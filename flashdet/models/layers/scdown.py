"""Spatial-Channel Decoupled Downsampling (YOLOv10)."""

import torch
import torch.nn as nn

from .conv import ConvBNSiLU


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
