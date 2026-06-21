"""
YOLOv11 — Next-generation YOLO with C3k2 blocks and enhanced SPPF.

Key innovations:
  - C3k2 block: C2f-style bottleneck with flexible 3×3 kernel support
  - Enhanced SPPF (Spatial Pyramid Pooling - Fast) with larger receptive field
  - Improved feature fusion with attention gating

Reference:
    Ultralytics YOLOv11, 2024.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.registry import BACKBONES

logger = logging.getLogger(__name__)


class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck with optional shortcut."""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, k: Tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        mid = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, mid, k[0])
        self.cv2 = ConvBNSiLU(mid, out_ch, k[1])
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3k2(nn.Module):
    """C3k2 block — CSP bottleneck with 2 convolutions and flexible kernels.

    A variant of C2f that uses Bottleneck blocks with configurable kernel
    sizes (defaulting to 3×3), enabling richer multi-scale feature extraction.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        k: int = 3,
    ):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch)
        self.blocks = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut=shortcut, k=(k, k))
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.blocks:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Applies sequential max-pooling with a fixed kernel to capture
    multi-scale context without significant compute overhead.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        mid = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, mid)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBNSiLU(mid * 4, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


class C2PSA(nn.Module):
    """C2f with Partial Self-Attention for YOLOv11.

    Integrates lightweight self-attention into the C2f architecture
    for improved global context modelling.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch)

        self.attn_blocks = nn.ModuleList()
        for _ in range(n):
            self.attn_blocks.append(nn.Sequential(
                ConvBNSiLU(self.c, self.c, 3),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.c, self.c),
                nn.SiLU(inplace=True),
                nn.Linear(self.c, self.c),
                nn.Sigmoid(),
            ))
        self.bottlenecks = nn.ModuleList([
            Bottleneck(self.c, self.c) for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for attn, bn in zip(self.attn_blocks, self.bottlenecks):
            feat = bn(y[-1])
            B, C, H, W = feat.shape
            scale = attn(feat).view(B, C, 1, 1)
            y.append(feat * scale)
        return self.cv2(torch.cat(y, 1))


class YOLOv11Neck(nn.Module):
    """PAN-FPN neck for YOLOv11 using C3k2 blocks."""

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvBNSiLU(ch, out_channels) for ch in in_channels])

        self.td_blocks = nn.ModuleList()
        self.bu_downs = nn.ModuleList()
        self.bu_blocks = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.td_blocks.append(C3k2(out_channels * 2, out_channels, n=2))
            self.bu_downs.append(ConvBNSiLU(out_channels, out_channels, 3, 2))
            self.bu_blocks.append(C3k2(out_channels * 2, out_channels, n=2))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        lats = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(lats) - 1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i - 1].shape[2:], mode="nearest")
            lats[i - 1] = self.td_blocks[i - 1](torch.cat([lats[i - 1], up], dim=1))

        for i in range(len(lats) - 1):
            down = self.bu_downs[i](lats[i])
            lats[i + 1] = self.bu_blocks[i](torch.cat([down, lats[i + 1]], dim=1))

        return lats


class YOLOv11Head(nn.Module):
    """Decoupled detection head for YOLOv11."""

    def __init__(self, num_classes: int, in_channels: int, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_convs = nn.Sequential(ConvBNSiLU(in_channels, in_channels, 3), ConvBNSiLU(in_channels, in_channels, 3))
        self.reg_convs = nn.Sequential(ConvBNSiLU(in_channels, in_channels, 3), ConvBNSiLU(in_channels, in_channels, 3))
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out = self.cls_pred(self.cls_convs(x))
        reg_out = self.reg_pred(self.reg_convs(x))
        return torch.cat([cls_out, reg_out], dim=1)


@BACKBONES.register("YOLOv11")
class YOLOv11(nn.Module):
    """YOLOv11 object detector with C3k2 blocks and SPPF.

    Args:
        num_classes: Number of detection classes.
        width_mult: Channel width multiplier (0.25=n, 0.5=s, 1.0=m, 1.25=l, 1.5=x).
        depth_mult: Block depth multiplier.
        reg_max: DFL regression max value.
        use_c2psa: Use C2PSA attention in the last backbone stage.
    """

    def __init__(
        self,
        num_classes: int = 80,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        reg_max: int = 16,
        use_c2psa: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        base = [64, 128, 256, 512]
        channels = [max(int(c * width_mult), 16) for c in base]
        n_blocks = [max(int(3 * depth_mult), 1) for _ in range(4)]

        # Backbone
        self.stem = ConvBNSiLU(3, channels[0], 3, 2)

        self.stage1 = nn.Sequential(
            ConvBNSiLU(channels[0], channels[1], 3, 2),
            C3k2(channels[1], channels[1], n=n_blocks[0]),
        )
        self.stage2 = nn.Sequential(
            ConvBNSiLU(channels[1], channels[2], 3, 2),
            C3k2(channels[2], channels[2], n=n_blocks[1]),
        )

        stage3_layers: List[nn.Module] = [
            ConvBNSiLU(channels[2], channels[3], 3, 2),
            C3k2(channels[3], channels[3], n=n_blocks[2]),
            SPPF(channels[3], channels[3]),
        ]
        if use_c2psa:
            stage3_layers.append(C2PSA(channels[3], channels[3], n=1))
        self.stage3 = nn.Sequential(*stage3_layers)

        # Neck
        neck_in = [channels[1], channels[2], channels[3]]
        neck_out = channels[2]
        self.neck = YOLOv11Neck(neck_in, neck_out)

        # Heads
        self.heads = nn.ModuleList([
            YOLOv11Head(num_classes, neck_out, reg_max) for _ in range(3)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for head in self.heads:
            nn.init.constant_(head.cls_pred.bias, -4.595)

    def forward(self, x: torch.Tensor, gt_meta: Optional[Dict] = None, **kwargs) -> Dict:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)

        neck_feats = self.neck([c3, c4, c5])
        preds = [head(f) for head, f in zip(self.heads, neck_feats)]

        return {"preds": preds}

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "YOLOv11",
            "num_classes": self.num_classes,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
