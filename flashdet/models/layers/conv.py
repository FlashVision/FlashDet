"""Shared convolution building blocks.

ConvBlock is a convenience alias for ConvModule with SiLU activation
and short parameter names. Both share the same internal structure.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + SiLU activation block.

    Short-form API for use in heads, necks, and blocks where SiLU
    is the standard activation. For configurable activation, use
    ConvModule directly.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))
