"""Spatial Pyramid Pooling - Fast."""

import torch
import torch.nn as nn

from .conv import ConvBNSiLU


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
