"""YOLOv9 PANet-style FPN neck.

Independent implementation based on:
    Wang et al., "YOLOv9: Learning What You Want to Learn Using
    Programmable Gradient Information", arXiv:2402.13616, 2024.

This is a clean-room implementation. No code copied from AGPL/GPL sources.
License: MIT (same as FlashDet)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.models.layers.yolo_blocks import ConvBNSiLU, GELAN


class YOLOv9Neck(nn.Module):
    """PANet-style FPN neck for YOLOv9."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvBNSiLU(ch, out_channels) for ch in in_channels])
        self.top_down_blocks = nn.ModuleList()
        self.bottom_up_convs = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.top_down_blocks.append(GELAN(out_channels * 2, out_channels, num_blocks=2))
            self.bottom_up_convs.append(ConvBNSiLU(out_channels, out_channels, 3, 2))
            self.bottom_up_blocks.append(GELAN(out_channels * 2, out_channels, num_blocks=2))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")
            laterals[i - 1] = self.top_down_blocks[i - 1](torch.cat([laterals[i - 1], up], dim=1))

        for i in range(len(laterals) - 1):
            down = self.bottom_up_convs[i](laterals[i])
            laterals[i + 1] = self.bottom_up_blocks[i](torch.cat([down, laterals[i + 1]], dim=1))

        return laterals
