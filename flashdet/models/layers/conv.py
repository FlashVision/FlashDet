"""Shared convolution building blocks."""

import torch
import torch.nn as nn


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


class DownSample(nn.Module):
    """Downsampling via 3x3 stride-2 conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNSiLU(in_ch, out_ch, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
