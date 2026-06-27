"""YOLOX backbone — Modified CSPDarknet with Focus and SPP.

Independent implementation based on:
    Ge et al., "YOLOX: Exceeding YOLO Series in 2021",
    arXiv:2107.08430, 2021.
    Original code: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0)

License: MIT (same as FlashDet)
"""

from typing import List

import torch
import torch.nn as nn

from flashdet.models.layers.yolo_blocks import ConvBNSiLU, Bottleneck


class Focus(nn.Module):
    """Focus module — slices input into 4 parts and concatenates (space-to-depth)."""

    def __init__(self, c1: int, c2: int, k: int = 1):
        super().__init__()
        self.conv = ConvBNSiLU(c1 * 4, c2, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1))


class CSPBlock(nn.Module):
    """Cross-Stage Partial block (YOLOX style)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvBNSiLU(c1, c_, 1)
        self.cv2 = ConvBNSiLU(c1, c_, 1)
        self.cv3 = ConvBNSiLU(2 * c_, c2, 1)
        self.blocks = nn.Sequential(*[Bottleneck(c_, c_, shortcut=shortcut) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], 1))


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling with multiple kernel sizes."""

    def __init__(self, c1: int, c2: int, kernels: tuple = (5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNSiLU(c1, c_, 1)
        self.cv2 = ConvBNSiLU(c_ * (len(kernels) + 1), c2, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [p(x) for p in self.pools], 1))


class YOLOXBackbone(nn.Module):
    """YOLOX CSPDarknet backbone with Focus stem and SPP.

    Args:
        width_mult: Channel multiplier (0.33=nano, 0.50=tiny, 0.75=s, 1.0=m/l).
        depth_mult: Depth multiplier (0.33=nano/tiny/s, 0.67=m, 1.0=l).
    """

    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0):
        super().__init__()
        base_channels = [64, 128, 256, 512, 1024]
        ch = [max(int(c * width_mult), 16) for c in base_channels]
        n = max(round(3 * depth_mult), 1)

        # Stem (Focus or Conv stride 2 for smaller models)
        self.stem = Focus(3, ch[0], 3)

        # Dark2 (stride 4)
        self.dark2 = nn.Sequential(
            ConvBNSiLU(ch[0], ch[1], 3, 2),
            CSPBlock(ch[1], ch[1], n=n),
        )
        # Dark3 (stride 8) → P3
        self.dark3 = nn.Sequential(
            ConvBNSiLU(ch[1], ch[2], 3, 2),
            CSPBlock(ch[2], ch[2], n=n * 3),
        )
        # Dark4 (stride 16) → P4
        self.dark4 = nn.Sequential(
            ConvBNSiLU(ch[2], ch[3], 3, 2),
            CSPBlock(ch[3], ch[3], n=n * 3),
        )
        # Dark5 (stride 32) → P5
        self.dark5 = nn.Sequential(
            ConvBNSiLU(ch[3], ch[4], 3, 2),
            CSPBlock(ch[4], ch[4], n=n),
            SPPBottleneck(ch[4], ch[4]),
        )

        self.out_channels = [ch[2], ch[3], ch[4]]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        p3 = self.dark3(x)    # stride 8
        p4 = self.dark4(p3)   # stride 16
        p5 = self.dark5(p4)   # stride 32
        return [p3, p4, p5]
