"""YOLOv9 backbone — GELAN-based feature extractor."""

from typing import List

import torch
import torch.nn as nn

from flashdet.models.layers import ConvBNSiLU, DownSample, GELAN


class YOLOv9Backbone(nn.Module):
    """GELAN-based backbone for YOLOv9.

    Produces multi-scale features at stages 1, 2, 3 for the FPN neck.

    Args:
        in_channels: Input image channels.
        width_mult: Channel width multiplier.
        depth_mult: Block depth multiplier.
    """

    def __init__(self, in_channels: int = 3, width_mult: float = 1.0, depth_mult: float = 1.0):
        super().__init__()
        base_channels = [64, 128, 256, 512]
        channels = [max(int(c * width_mult), 16) for c in base_channels]
        depths = [max(int(3 * depth_mult), 1) for _ in range(4)]

        self.stem = nn.Sequential(
            ConvBNSiLU(in_channels, channels[0], 3, 2),
            ConvBNSiLU(channels[0], channels[0], 3),
        )
        self.stage1 = nn.Sequential(DownSample(channels[0], channels[1]), GELAN(channels[1], channels[1], num_blocks=depths[0]))
        self.stage2 = nn.Sequential(DownSample(channels[1], channels[2]), GELAN(channels[2], channels[2], num_blocks=depths[1]))
        self.stage3 = nn.Sequential(DownSample(channels[2], channels[3]), GELAN(channels[3], channels[3], num_blocks=depths[2]))

        self.out_channels = [channels[1], channels[2], channels[3]]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]
