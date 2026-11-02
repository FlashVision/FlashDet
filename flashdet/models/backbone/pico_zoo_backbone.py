"""Pico-Zoo backbone — one ultra-light backbone per architecture family.

This module exposes every architecture name from the Pico model table as a
selectable backbone.  All variants share the same deployment-friendly macro
layout (stem /4, then three stages at strides 8, 16 and 32) so they are drop-in
replacements for :class:`PicoBackbone` and the existing Pico neck/head.

The block implementation follows the architecture *family* rather than
reproducing every paper exactly:

======  ==================================================================
Family  Representative architectures
======  ==================================================================
rep     RepViT, MobileOne, FastViT, RepGhostNet, UniRepLKNet, EfficientRep
attn    EdgeNeXt, MobileViTv2, EfficientFormerV2, SHViT, EfficientViT,
        SwiftFormer, MobileFormer, EdgeViT, TinyViT, LeViT, LowFormer
ghost   EfficientNetV2, EMO, FasterNet, StarNet, MobileNetV4, GhostNetV2,
        ConvNeXtV2
mamba   MobileMamba, EfficientVMamba, MambaVision, MambaOut, EfficientViM,
        LocalMamba, VMamba, PlainMamba
lcnet   PP-LCNet, LCNet, ESNet, PP-LCNetv2, PeleeNet, ShuffleNetV2, ReXNet
======  ==================================================================

Using a shared layout keeps parameter counts in the Pico range and guarantees
that the backbone works with 320/416 inputs on CPU and edge devices.  The
``mamba`` family uses a MambaOut-style gated depthwise mixer, so no CUDA-only
selective-scan kernel is required.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from flashdet.models.layers.reparam import StrideDown, make_pico_block

# name -> (family, stem_channels, stage_depths)
# Stage channels are derived as (2*stem, 4*stem, 8*stem) so every variant
# keeps the same 3-level FPN interface.
PICO_BACKBONE_PRESETS: Dict[str, Tuple[str, int, Tuple[int, int, int]]] = {
    # A. Reparameterized CNN
    "repvit": ("rep", 24, (2, 3, 1)),
    "mobileone": ("rep", 32, (2, 3, 1)),
    "fastvit": ("rep", 24, (2, 4, 2)),
    "repghostnet": ("ghost", 20, (2, 3, 1)),
    # B. Lightweight attention
    "edgenext": ("attention", 24, (2, 3, 1)),
    "mobilevitv2": ("attention", 28, (1, 2, 1)),
    "efficientformerv2": ("attention", 24, (2, 2, 1)),
    "shvit": ("attention", 24, (1, 3, 1)),
    "efficientvit": ("attention", 32, (1, 2, 1)),
    "swiftformer": ("attention", 24, (2, 3, 1)),
    # C. Lightweight CNN
    "efficientnetv2": ("ghost", 28, (2, 4, 2)),
    "emo": ("attention", 32, (2, 3, 1)),
    "fasternet": ("ghost", 24, (2, 3, 1)),
    "starnet": ("ghost", 16, (2, 3, 1)),
    "mobilenetv4": ("ghost", 24, (2, 3, 1)),
    "ghostnetv2": ("ghost", 24, (2, 3, 1)),
    # D. Visual Mamba / SSM
    "mobilemamba": ("mamba", 24, (1, 2, 1)),
    "efficientvmamba": ("mamba", 24, (1, 2, 1)),
    "mambavision": ("mamba", 32, (1, 2, 1)),
    "mambaout": ("mamba", 24, (2, 3, 1)),
    "efficientvim": ("mamba", 24, (1, 2, 1)),
    # E. Detection-oriented CNN
    "pp_lcnet": ("lcnet", 16, (3, 4, 2)),
    "lcnet": ("lcnet", 16, (3, 4, 2)),
    "esnet": ("lcnet", 24, (2, 3, 1)),
    "pp_lcnetv2": ("lcnet", 24, (3, 4, 2)),
    "peleenet": ("ghost", 24, (3, 4, 2)),
    "shufflenetv2": ("lcnet", 24, (2, 4, 2)),
    # F. Modern efficient CNN
    "inceptionnext": ("rep", 24, (2, 3, 1)),
    "unireplknet": ("rep", 32, (2, 3, 1)),
    "convnextv2": ("ghost", 24, (1, 2, 1)),
    "efficientrep": ("rep", 24, (3, 4, 2)),
    "rexnet": ("lcnet", 24, (2, 3, 1)),
    # G. Lightweight transformers
    "mobileformer": ("attention", 24, (1, 2, 1)),
    "edgevit": ("attention", 24, (1, 3, 1)),
    "tinyvit": ("attention", 32, (1, 2, 1)),
    "levit": ("attention", 24, (2, 3, 1)),
    "lowformer": ("attention", 24, (1, 2, 1)),
    # H. Additional SSM
    "localmamba": ("mamba", 24, (1, 2, 1)),
    "vmamba": ("mamba", 32, (1, 2, 1)),
    "plainmamba": ("mamba", 24, (1, 2, 1)),
}

PICO_BACKBONE_ALIASES = {
    "pico": "repvit",
    "pico_v2": "repvit",
    "pico-v2": "repvit",
    "repnext": "repvit",
    "repnext_v2": "repvit",
    "mobilenet_v4": "mobilenetv4",
    "efficientnet_v2": "efficientnetv2",
    "ghostnet_v2": "ghostnetv2",
    "mobile_vitv2": "mobilevitv2",
    "efficientformer_v2": "efficientformerv2",
    "inception_next": "inceptionnext",
    "unirep_lknet": "unireplknet",
    "convnext_v2": "convnextv2",
    "tiny_vit": "tinyvit",
    "pplcnet": "pp_lcnet",
    "pp_lcnet_v2": "pp_lcnetv2",
    "shufflenet": "shufflenetv2",
    "shufflenet_v2": "shufflenetv2",
    "mamba_out": "mambaout",
    "efficient_vim": "efficientvim",
    "efficient_vmamba": "efficientvmamba",
    "mamba_vision": "mambavision",
    "mobile_mamba": "mobilemamba",
    "local_mamba": "localmamba",
    "plain_mamba": "plainmamba",
    "low_former": "lowformer",
    "edge_vit": "edgevit",
    "mobile_former": "mobileformer",
    "efficient_vit": "efficientvit",
}

PICO_BACKBONE_NAMES: Tuple[str, ...] = tuple(sorted(PICO_BACKBONE_PRESETS))


def normalize_backbone_name(name: str) -> str:
    """Normalize user input to a preset key."""
    key = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    key = PICO_BACKBONE_ALIASES.get(key, key)
    if key not in PICO_BACKBONE_PRESETS:
        raise ValueError(
            f"Unknown Pico backbone '{name}'. "
            f"Supported: {', '.join(PICO_BACKBONE_NAMES)}"
        )
    return key


def get_pico_preset(name: str) -> Dict:
    """Return the resolved preset dictionary for a backbone name."""
    key = normalize_backbone_name(name)
    family, stem, depths = PICO_BACKBONE_PRESETS[key]
    return {
        "name": key,
        "family": family,
        "stem_channels": stem,
        "stage_depths": tuple(depths),
        "stage_channels": (stem * 2, stem * 4, stem * 8),
    }


class PicoZooBackbone(nn.Module):
    """Ultra-light multi-scale backbone for any Pico architecture name.

    Args:
        arch: Architecture name (e.g. ``repvit``, ``edgenext``, ``mambaout``,
            ``pp_lcnet``).  Aliases such as ``mobilenet_v4`` are accepted.
        stem_channels: Optional override for the stem width.
        stage_depths: Optional override for per-stage block counts.
        out_stages: Which stages to return (default all three).
        activation: Activation name.
        pretrained: Accepted for API compatibility.  Pico-Zoo variants are
            trained from scratch; ImageNet weights are not downloaded.
    """

    def __init__(
        self,
        arch: str = "repvit",
        stem_channels: int = None,
        stage_depths: Tuple[int, ...] = None,
        out_stages: Tuple[int, ...] = (0, 1, 2),
        activation: str = "LeakyReLU",
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        preset = get_pico_preset(arch)
        self.arch = preset["name"]
        self.family = preset["family"]
        self.pretrained_loaded = False
        self.out_stages = tuple(out_stages)

        stem_ch = int(stem_channels or preset["stem_channels"])
        depths = tuple(stage_depths or preset["stage_depths"])
        if len(depths) != 3:
            raise ValueError("stage_depths must contain exactly 3 values")
        stage_channels = (stem_ch * 2, stem_ch * 4, stem_ch * 8)
        self.out_channels = [stage_channels[s] for s in self.out_stages]

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)

        in_ch = stem_ch
        self.stages = nn.ModuleList()
        for ch, depth in zip(stage_channels, depths):
            layers: List[nn.Module] = [StrideDown(in_ch)]
            for _ in range(max(1, int(depth))):
                layers.append(make_pico_block(self.family, ch, activation))
            self.stages.append(nn.Sequential(*layers))
            in_ch = ch

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.pool(self.stem(x))
        outputs: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_stages:
                outputs.append(x)
        return outputs

    @torch.no_grad()
    def fuse(self):
        """Fuse reparameterizable blocks (the ``rep`` family) for deployment."""
        def _fuse(module):
            for _, child in module.named_children():
                if child is self:
                    continue
                if hasattr(child, "fuse") and not isinstance(child, PicoZooBackbone):
                    child.fuse()
                else:
                    _fuse(child)

        _fuse(self)
        return self


def list_pico_backbones() -> List[str]:
    """Return all supported Pico backbone names."""
    return list(PICO_BACKBONE_NAMES)
