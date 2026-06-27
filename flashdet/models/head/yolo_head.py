"""Shared decoupled detection heads for YOLO-family models.

Independent implementation based on:
    Wang et al., "YOLOv7: Trainable bag-of-freebies sets new
    state-of-the-art for real-time object detectors", CVPR 2023.
"""

import torch
import torch.nn as nn

from flashdet.models.layers.yolo_blocks import ConvBNSiLU


class YOLODetectionHead(nn.Module):
    """Decoupled detection head shared by YOLOv9 and YOLOv11."""

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


class DualHeadOne2One(nn.Module):
    """One-to-one head for NMS-free inference (YOLOv10)."""

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
    """One-to-many head for richer supervision during training (YOLOv10)."""

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


class PGIAuxBranch(nn.Module):
    """Auxiliary reversible branch for Programmable Gradient Information (YOLOv9).

    Provides auxiliary supervision during training to preserve gradient
    information flow. Disabled at inference time.
    """

    def __init__(self, channels: list, num_classes: int):
        super().__init__()
        self.aux_heads = nn.ModuleList()
        for ch in channels:
            self.aux_heads.append(nn.Sequential(
                ConvBNSiLU(ch, ch, 3),
                ConvBNSiLU(ch, ch, 3),
                nn.Conv2d(ch, num_classes + 4, 1),
            ))

    def forward(self, features: list) -> list:
        return [head(f) for head, f in zip(self.aux_heads, features)]
