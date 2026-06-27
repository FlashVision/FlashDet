"""YOLOv8 backbone — CSPDarknet with C2f blocks.

Independent implementation based on:
    Jocher et al., "YOLOv8", Ultralytics, 2023.
    Architecture: CSPDarknet53 modernized with C2f (CSP Bottleneck 2 convs).

This is a clean-room implementation. No code copied from AGPL sources.
License: MIT (same as FlashDet)
"""

from typing import List

import torch
import torch.nn as nn

from flashdet.models.layers.yolo_blocks import ConvBNSiLU, C2f


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (single kernel size, 3 sequential pools)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNSiLU(c1, c_, 1)
        self.cv2 = ConvBNSiLU(c_ * 4, c2, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class YOLOv8Backbone(nn.Module):
    """CSPDarknet backbone for YOLOv8 with C2f blocks and SPPF.

    Produces 3-scale feature maps at strides 8, 16, 32.

    Args:
        width_mult: Channel width multiplier (0.25=n, 0.5=s, 0.75=m, 1.0=l).
        depth_mult: Depth multiplier for C2f blocks (0.33=n, 0.33=s, 0.67=m, 1.0=l).
    """

    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0):
        super().__init__()
        base_channels = [64, 128, 256, 512, 1024]
        ch = [max(int(c * width_mult), 16) for c in base_channels]
        n = max(round(3 * depth_mult), 1)

        # Stem + Stage 1 (stride 2)
        self.stem = ConvBNSiLU(3, ch[0], 3, 2)
        # Stage 2 (stride 4)
        self.stage1 = nn.Sequential(
            ConvBNSiLU(ch[0], ch[1], 3, 2),
            C2f(ch[1], ch[1], n=n, shortcut=True),
        )
        # Stage 3 (stride 8) → P3
        self.stage2 = nn.Sequential(
            ConvBNSiLU(ch[1], ch[2], 3, 2),
            C2f(ch[2], ch[2], n=n, shortcut=True),
        )
        # Stage 4 (stride 16) → P4
        self.stage3 = nn.Sequential(
            ConvBNSiLU(ch[2], ch[3], 3, 2),
            C2f(ch[3], ch[3], n=n, shortcut=True),
        )
        # Stage 5 (stride 32) → P5
        self.stage4 = nn.Sequential(
            ConvBNSiLU(ch[3], ch[4], 3, 2),
            C2f(ch[4], ch[4], n=n, shortcut=True),
            SPPF(ch[4], ch[4]),
        )

        self.out_channels = [ch[2], ch[3], ch[4]]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)    # stride 8
        p4 = self.stage3(p3)   # stride 16
        p5 = self.stage4(p4)   # stride 32
        return [p3, p4, p5]
