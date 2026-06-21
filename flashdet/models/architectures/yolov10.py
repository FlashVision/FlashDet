"""
YOLOv10 — NMS-Free Real-Time End-to-End Object Detection.

Key innovations:
  - Dual label assignment: one-to-many for training, one-to-one for inference
  - Consistent dual assignment for NMS-free end-to-end detection
  - Efficiency-driven model design with spatial-channel decoupled downsampling

Reference:
    Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", 2024.
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


class SCDown(nn.Module):
    """Spatial-Channel Decoupled Downsampling.

    Reduces spatial dimensions and adjusts channels in separate steps
    for improved efficiency.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 2):
        super().__init__()
        self.pointwise = ConvBNSiLU(in_ch, out_ch, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, k, s, k // 2, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.pointwise(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions — standard YOLO building block."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(self.c, self.c, 3),
                ConvBNSiLU(self.c, self.c, 3),
            )
            for _ in range(n)
        ])
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.blocks:
            out = blk(y[-1])
            if self.shortcut and out.shape == y[-1].shape:
                out = out + y[-1]
            y.append(out)
        return self.cv2(torch.cat(y, 1))


class PSA(nn.Module):
    """Partial Self-Attention module for efficient global context."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)

        return x + self.norm(self.proj(out))


class DualHeadOne2One(nn.Module):
    """One-to-one head for NMS-free inference (picks the single best prediction
    per ground-truth during training with a simulated argmax)."""

    def __init__(self, num_classes: int, in_channels: int, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_conv = nn.Sequential(ConvBNSiLU(in_channels, in_channels, 3), ConvBNSiLU(in_channels, in_channels, 3))
        self.reg_conv = nn.Sequential(ConvBNSiLU(in_channels, in_channels, 3), ConvBNSiLU(in_channels, in_channels, 3))
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out = self.cls_pred(self.cls_conv(x))
        reg_out = self.reg_pred(self.reg_conv(x))
        return torch.cat([cls_out, reg_out], dim=1)


class DualHeadOne2Many(nn.Module):
    """One-to-many head for richer supervision during training."""

    def __init__(self, num_classes: int, in_channels: int, reg_max: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_conv = nn.Sequential(ConvBNSiLU(in_channels, in_channels, 3), ConvBNSiLU(in_channels, in_channels, 3))
        self.reg_conv = nn.Sequential(ConvBNSiLU(in_channels, in_channels, 3), ConvBNSiLU(in_channels, in_channels, 3))
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_pred = nn.Conv2d(in_channels, 4 * (reg_max + 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out = self.cls_pred(self.cls_conv(x))
        reg_out = self.reg_pred(self.reg_conv(x))
        return torch.cat([cls_out, reg_out], dim=1)


class YOLOv10Neck(nn.Module):
    """Efficient PANet neck for YOLOv10."""

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvBNSiLU(ch, out_channels) for ch in in_channels])
        self.td_blocks = nn.ModuleList()
        self.bu_downs = nn.ModuleList()
        self.bu_blocks = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.td_blocks.append(C2f(out_channels * 2, out_channels, n=2))
            self.bu_downs.append(SCDown(out_channels, out_channels))
            self.bu_blocks.append(C2f(out_channels * 2, out_channels, n=2))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        lats = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(lats) - 1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i - 1].shape[2:], mode="nearest")
            lats[i - 1] = self.td_blocks[i - 1](torch.cat([lats[i - 1], up], dim=1))

        for i in range(len(lats) - 1):
            down = self.bu_downs[i](lats[i])
            lats[i + 1] = self.bu_blocks[i](torch.cat([down, lats[i + 1]], dim=1))

        return lats


@BACKBONES.register("YOLOv10")
class YOLOv10(nn.Module):
    """YOLOv10 detector with NMS-free end-to-end design.

    Uses consistent dual assignment: one-to-many head for training supervision
    and one-to-one head for NMS-free inference.

    Args:
        num_classes: Number of detection classes.
        width_mult: Channel width multiplier.
        depth_mult: Depth multiplier for bottleneck blocks.
        reg_max: DFL regression max value.
        use_psa: Use Partial Self-Attention in backbone.
    """

    def __init__(
        self,
        num_classes: int = 80,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        reg_max: int = 16,
        use_psa: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        base = [64, 128, 256, 512]
        channels = [max(int(c * width_mult), 16) for c in base]
        n_blocks = [max(int(3 * depth_mult), 1) for _ in range(4)]

        self.stem = ConvBNSiLU(3, channels[0], 3, 2)

        self.stage1 = nn.Sequential(SCDown(channels[0], channels[1]), C2f(channels[1], channels[1], n_blocks[0]))
        self.stage2 = nn.Sequential(SCDown(channels[1], channels[2]), C2f(channels[2], channels[2], n_blocks[1]))

        stage3_modules = [SCDown(channels[2], channels[3]), C2f(channels[3], channels[3], n_blocks[2])]
        if use_psa:
            stage3_modules.append(PSA(channels[3]))
        self.stage3 = nn.Sequential(*stage3_modules)

        neck_in = [channels[1], channels[2], channels[3]]
        neck_out = channels[2]
        self.neck = YOLOv10Neck(neck_in, neck_out)

        self.o2o_heads = nn.ModuleList([
            DualHeadOne2One(num_classes, neck_out, reg_max) for _ in range(3)
        ])
        self.o2m_heads = nn.ModuleList([
            DualHeadOne2Many(num_classes, neck_out, reg_max) for _ in range(3)
        ])

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

        neck_feats = self.neck([c3, c4, c5])

        o2o_preds = [h(f) for h, f in zip(self.o2o_heads, neck_feats)]
        result: Dict = {"preds": o2o_preds}

        if self.training:
            o2m_preds = [h(f) for h, f in zip(self.o2m_heads, neck_feats)]
            result["o2m_preds"] = o2m_preds

        return result

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "YOLOv10",
            "num_classes": self.num_classes,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
