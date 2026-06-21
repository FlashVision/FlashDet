"""
YOLOv9 — Programmable Gradient Information (PGI) with GELAN architecture.

Key innovations:
  - GELAN (Generalized Efficient Layer Aggregation Network) for the backbone
  - PGI (Programmable Gradient Information) with auxiliary reversible branch

Reference:
    Wang et al., "YOLOv9: Learning What You Want to Learn Using
    Programmable Gradient Information", 2024.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.registry import BACKBONES

logger = logging.getLogger(__name__)


class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU activation block."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """Re-parameterisable convolution (3x3 + 1x1 branches, fused at inference)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_ch, out_ch, k, s)
        self.conv2 = ConvBNSiLU(in_ch, out_ch, 1, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.conv2(x)


class ELAN(nn.Module):
    """Efficient Layer Aggregation Network block.

    Splits input into two branches: one passes through a series of
    convolutions with intermediate concatenations for gradient-rich
    aggregation.
    """

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, num_blocks: int = 2):
        super().__init__()
        mid_ch = mid_ch or out_ch // 2
        self.conv1 = ConvBNSiLU(in_ch, mid_ch)
        self.conv2 = ConvBNSiLU(in_ch, mid_ch)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                ConvBNSiLU(mid_ch, mid_ch, 3),
                ConvBNSiLU(mid_ch, mid_ch, 3),
            ))

        self.merge = ConvBNSiLU(mid_ch * (2 + num_blocks), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        parts = [x1, x2]
        cur = x2
        for blk in self.blocks:
            cur = blk(cur)
            parts.append(cur)
        return self.merge(torch.cat(parts, dim=1))


class GELAN(nn.Module):
    """Generalized ELAN — YOLOv9 backbone based on ELAN with flexible routing."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, num_blocks: int = 3):
        super().__init__()
        mid_ch = mid_ch or out_ch // 2
        self.conv1 = ConvBNSiLU(in_ch, mid_ch)
        self.conv2 = ConvBNSiLU(in_ch, mid_ch)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                ConvBNSiLU(mid_ch, mid_ch, 3),
                ConvBNSiLU(mid_ch, mid_ch, 3),
            ))

        total_concat = mid_ch * (2 + num_blocks)
        self.transition = ConvBNSiLU(total_concat, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        outs = [x1, x2]
        cur = x2
        for blk in self.blocks:
            cur = blk(cur)
            outs.append(cur)
        return self.transition(torch.cat(outs, dim=1))


class DownSample(nn.Module):
    """Downsampling via 3x3 stride-2 conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNSiLU(in_ch, out_ch, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PGIAuxBranch(nn.Module):
    """Auxiliary reversible branch for Programmable Gradient Information.

    Provides auxiliary supervision during training to preserve gradient
    information flow. Disabled at inference time.
    """

    def __init__(self, channels: List[int], num_classes: int):
        super().__init__()
        self.aux_heads = nn.ModuleList()
        for ch in channels:
            self.aux_heads.append(nn.Sequential(
                ConvBNSiLU(ch, ch, 3),
                ConvBNSiLU(ch, ch, 3),
                nn.Conv2d(ch, num_classes + 4, 1),
            ))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [head(f) for head, f in zip(self.aux_heads, features)]


class YOLOv9Neck(nn.Module):
    """PANet-style FPN neck for YOLOv9."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvBNSiLU(ch, out_channels) for ch in in_channels])
        self.top_down_blocks = nn.ModuleList()
        self.bottom_up_convs = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.top_down_blocks.append(GELAN(out_channels * 2, out_channels, num_blocks=2))
            self.bottom_up_convs.append(ConvBNSiLU(out_channels, out_channels, 3, 2))
            self.bottom_up_blocks.append(GELAN(out_channels * 2, out_channels, num_blocks=2))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")
            laterals[i - 1] = self.top_down_blocks[i - 1](torch.cat([laterals[i - 1], up], dim=1))

        # Bottom-up
        for i in range(len(laterals) - 1):
            down = self.bottom_up_convs[i](laterals[i])
            laterals[i + 1] = self.bottom_up_blocks[i](torch.cat([down, laterals[i + 1]], dim=1))

        return laterals


class YOLOv9Head(nn.Module):
    """Decoupled detection head for YOLOv9."""

    def __init__(self, num_classes: int, in_channels: int = 256, num_anchors: int = 1, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_convs = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3),
            ConvBNSiLU(in_channels, in_channels, 3),
        )
        self.reg_convs = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3),
            ConvBNSiLU(in_channels, in_channels, 3),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out = self.cls_pred(self.cls_convs(x))
        reg_out = self.reg_pred(self.reg_convs(x))
        return torch.cat([cls_out, reg_out], dim=1)


@BACKBONES.register("YOLOv9")
class YOLOv9(nn.Module):
    """YOLOv9 object detector with GELAN backbone and PGI.

    Args:
        num_classes: Number of detection classes.
        width_mult: Channel width multiplier.
        depth_mult: Block depth multiplier.
        in_channels: Input image channels.
        reg_max: DFL regression max value.
        use_pgi: Enable PGI auxiliary branch during training.
    """

    def __init__(
        self,
        num_classes: int = 80,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        in_channels: int = 3,
        reg_max: int = 16,
        use_pgi: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_pgi = use_pgi

        base_channels = [64, 128, 256, 512]
        channels = [max(int(c * width_mult), 16) for c in base_channels]
        depths = [max(int(3 * depth_mult), 1) for _ in range(4)]

        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, channels[0], 3, 2),
            ConvBNSiLU(channels[0], channels[0], 3),
        )

        self.stage1 = nn.Sequential(DownSample(channels[0], channels[1]), GELAN(channels[1], channels[1], num_blocks=depths[0]))
        self.stage2 = nn.Sequential(DownSample(channels[1], channels[2]), GELAN(channels[2], channels[2], num_blocks=depths[1]))
        self.stage3 = nn.Sequential(DownSample(channels[2], channels[3]), GELAN(channels[3], channels[3], num_blocks=depths[2]))

        neck_in = [channels[1], channels[2], channels[3]]
        neck_out = channels[2]
        self.neck = YOLOv9Neck(neck_in, neck_out)

        self.heads = nn.ModuleList([
            YOLOv9Head(num_classes, neck_out, reg_max=reg_max)
            for _ in range(3)
        ])

        if use_pgi:
            self.pgi_aux = PGIAuxBranch(neck_in, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, gt_meta: Optional[Dict] = None, **kwargs) -> Dict:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)

        backbone_feats = [c3, c4, c5]
        neck_feats = self.neck(backbone_feats)

        preds = [head(f) for head, f in zip(self.heads, neck_feats)]

        result: Dict = {"preds": preds}

        if self.training and self.use_pgi:
            result["aux_preds"] = self.pgi_aux(backbone_feats)

        return result

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "YOLOv9",
            "num_classes": self.num_classes,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
