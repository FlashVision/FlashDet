"""Pico-Zoo necks — family-matched lightweight PAN/FPN feature fusion.

Every Pico backbone family has a matching neck block:

================  ==========================================
Backbone family   Neck block
================  ==========================================
rep               RepBlock (reparameterizable, Category A/F)
attention         AttentionBlock (Category B/G)
ghost             GhostBlock (Category C)
mamba             SSMBlock (Category D/H)
lcnet             LCNetBlock (Category E)
================  ==========================================

The neck keeps the same three-level PAN layout and output interface as
:class:`PicoNeck`, so it is a drop-in replacement for every Pico detector.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.models.layers.conv_module import ConvModule, DepthwiseConvModule
from flashdet.models.layers.reparam import make_pico_block
from flashdet.models.neck.pico_neck import DySample, PicoNeck
from flashdet.models.backbone.pico_zoo_backbone import (
    PICO_BACKBONE_PRESETS,
    normalize_backbone_name,
)

PICO_NECK_FAMILIES: Tuple[str, ...] = (
    "pan", "rep", "ghost", "attention", "mamba", "lcnet",
)

FAMILY_TO_NECK = {
    "rep": "rep",
    "attention": "attention",
    "ghost": "ghost",
    "mamba": "mamba",
    "lcnet": "lcnet",
}


def resolve_neck_family(neck_type: Optional[str]) -> str:
    """Resolve a neck family from a neck name or backbone architecture name."""
    if neck_type is None:
        return "pan"
    key = str(neck_type).strip().lower().replace("-", "_")
    if key in ("", "pan", "pico", "pico_neck", "fpn", "pafpn"):
        return "pan"
    if key in PICO_NECK_FAMILIES:
        return key
    # Accept any backbone name (e.g. --neck edgenext -> attention)
    try:
        backbone = normalize_backbone_name(key)
    except ValueError:
        raise ValueError(
            f"Unknown Pico neck '{neck_type}'. Supported neck families: "
            f"{', '.join(PICO_NECK_FAMILIES)}; or pass a supported backbone name."
        )
    return FAMILY_TO_NECK[PICO_BACKBONE_PRESETS[backbone][0]]


class FusionStage(nn.Module):
    """Project concatenated features, then apply N family blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        family: str,
        num_blocks: int = 1,
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        if in_channels == out_channels:
            self.reduce: nn.Module = nn.Identity()
        else:
            self.reduce = ConvModule(
                in_channels, out_channels, 1, activation=activation
            )
        self.blocks = nn.Sequential(*[
            make_pico_block(family, out_channels, activation)
            for _ in range(max(1, int(num_blocks)))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.reduce(x))


class PicoZooNeck(nn.Module):
    """Family-matched lightweight PAN neck for Pico-Zoo backbones."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 80,
        family: str = "rep",
        kernel_size: int = 5,
        num_blocks: int = 1,
        num_extra_level: int = 0,
        use_depthwise: bool = True,
        use_dysample: bool = True,
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.family = resolve_neck_family(family)

        Conv = DepthwiseConvModule if use_depthwise else ConvModule

        # Lateral 1x1 reductions
        self.reduce_layers = nn.ModuleList([
            ConvModule(ch, out_channels, 1, activation=activation)
            for ch in self.in_channels
        ])

        # Top-down upsamplers (DySample when the width permits, else bilinear)
        self.upsamplers = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            if use_dysample and out_channels % 4 == 0:
                self.upsamplers.append(DySample(out_channels, scale=2, groups=4))
            else:
                self.upsamplers.append(None)

        # Top-down fusion
        self.top_down_blocks = nn.ModuleList([
            FusionStage(out_channels * 2, out_channels, self.family,
                        num_blocks=num_blocks, activation=activation)
            for _ in range(len(self.in_channels) - 1)
        ])

        # Bottom-up fusion
        self.downsamples = nn.ModuleList([
            Conv(out_channels, out_channels, kernel_size, stride=2,
                 activation=activation)
            for _ in range(len(self.in_channels) - 1)
        ])
        self.bottom_up_blocks = nn.ModuleList([
            FusionStage(out_channels * 2, out_channels, self.family,
                        num_blocks=num_blocks, activation=activation)
            for _ in range(len(self.in_channels) - 1)
        ])

        self.num_extra_level = num_extra_level
        self.extra_lvl_in_conv = nn.ModuleList()
        self.extra_lvl_out_conv = nn.ModuleList()
        for _ in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                Conv(out_channels, out_channels, kernel_size, stride=2,
                     activation=activation)
            )
            self.extra_lvl_out_conv.append(
                Conv(out_channels, out_channels, kernel_size, stride=2,
                     activation=activation)
            )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(inputs) == len(self.in_channels)

        inputs = [red(x) for red, x in zip(self.reduce_layers, inputs)]

        # Top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            up_idx = len(self.in_channels) - 1 - idx

            upsampler = self.upsamplers[up_idx]
            if upsampler is not None:
                upsample_feat = upsampler(feat_high)
            else:
                upsample_feat = F.interpolate(
                    feat_high, size=feat_low.shape[2:],
                    mode="bilinear", align_corners=False,
                )
            # Always size-match the low-resolution feature. This is a no-op
            # when the shapes already agree, and keeping it unconditional lets
            # torch.fx trace the forward without data-dependent control flow
            # (needed for graph fusion and export).
            upsample_feat = F.interpolate(
                upsample_feat, size=feat_low.shape[2:],
                mode="bilinear", align_corners=False,
            )

            inner_out = self.top_down_blocks[up_idx](
                torch.cat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        # Bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_high], dim=1)
            )
            outs.append(out)

        for extra_in, extra_out in zip(
            self.extra_lvl_in_conv, self.extra_lvl_out_conv
        ):
            outs.append(extra_in(inputs[-1]) + extra_out(outs[-1]))

        return outs


def build_pico_neck(
    neck_type: Optional[str],
    in_channels: List[int],
    out_channels: int = 80,
    **kwargs,
) -> nn.Module:
    """Build a Pico neck from a neck family or architecture name.

    ``pan`` (the default) returns the original :class:`PicoNeck`; all other
    families return :class:`PicoZooNeck`.
    """
    family = resolve_neck_family(neck_type)
    if family == "pan":
        # PicoNeck does not know zoo-only options such as use_dysample.
        allowed = {
            "kernel_size", "expand", "num_blocks", "use_res",
            "num_extra_level", "use_depthwise", "activation",
        }
        pan_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return PicoNeck(
            in_channels=in_channels,
            out_channels=out_channels,
            **pan_kwargs,
        )
    return PicoZooNeck(
        in_channels=in_channels,
        out_channels=out_channels,
        family=family,
        **kwargs,
    )


def list_pico_necks() -> List[str]:
    """Return all supported Pico neck families."""
    return list(PICO_NECK_FAMILIES)
