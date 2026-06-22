"""Re-parameterisable convolution blocks."""

import torch
import torch.nn as nn

from .conv import ConvBNSiLU


class RepConv(nn.Module):
    """Re-parameterisable convolution (3x3 + 1x1 branches, fused at inference)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_ch, out_ch, k, s)
        self.conv2 = ConvBNSiLU(in_ch, out_ch, 1, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.conv2(x)


class RepVGGBlock(nn.Module):
    """RepVGG-style block with re-parameterisable 3x3+1x1+identity branches."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_identity = (stride == 1 and in_channels == out_channels)
        if self.use_identity:
            self.bn_identity = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.use_identity:
            out = out + self.bn_identity(x)
        return self.act(out)
