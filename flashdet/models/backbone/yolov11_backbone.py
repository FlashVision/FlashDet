"""YOLOv11 backbone — C2f variant with enhanced feature extraction.

Independent implementation based on:
    Jocher et al., YOLO11, Ultralytics, 2024.

This is a clean-room implementation. No code copied from AGPL/GPL sources.
License: MIT (same as FlashDet)
"""

from typing import List

import torch
import torch.nn as nn

from flashdet.models.layers.yolo_blocks import ConvBNSiLU, C2f, SCDown, PSA


class YOLOv11Backbone(nn.Module):
    """C2f-based backbone for YOLOv11 (C3k2 blocks + optional C2PSA).

    Args:
        width_mult: Channel width multiplier.
        depth_mult: Depth multiplier for bottleneck blocks.
        use_c2psa: Use C2PSA attention in the last stage.
    """

    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0, use_c2psa: bool = True):
        super().__init__()
        base = [64, 128, 256, 512]
        channels = [max(int(c * width_mult), 16) for c in base]
        n_blocks = [max(int(3 * depth_mult), 1) for _ in range(4)]

        self.stem = ConvBNSiLU(3, channels[0], 3, 2)
        self.stage1 = nn.Sequential(SCDown(channels[0], channels[1]), C2f(channels[1], channels[1], n_blocks[0]))
        self.stage2 = nn.Sequential(SCDown(channels[1], channels[2]), C2f(channels[2], channels[2], n_blocks[1]))

        stage3_modules: List[nn.Module] = [SCDown(channels[2], channels[3]), C2f(channels[3], channels[3], n_blocks[2])]
        if use_c2psa:
            stage3_modules.append(PSA(channels[3]))
        self.stage3 = nn.Sequential(*stage3_modules)

        self.out_channels = [channels[1], channels[2], channels[3]]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]
