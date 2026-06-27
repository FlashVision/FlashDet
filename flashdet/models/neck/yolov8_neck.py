"""YOLOv8 PANet-style FPN neck with C2f blocks.

Independent implementation based on:
    Jocher et al., "YOLOv8", Ultralytics, 2023.

This is a clean-room implementation. No code copied from AGPL sources.
License: MIT (same as FlashDet)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.models.layers.yolo_blocks import ConvBNSiLU, C2f


class YOLOv8Neck(nn.Module):
    """PANet FPN neck for YOLOv8 using C2f blocks.

    Top-down path reduces channels, bottom-up path builds multi-scale features.

    Args:
        in_channels: Channel sizes from backbone [P3, P4, P5].
        depth_mult: Depth multiplier for C2f blocks.
    """

    def __init__(self, in_channels: List[int], depth_mult: float = 1.0):
        super().__init__()
        c3, c4, c5 = in_channels
        n = max(round(3 * depth_mult), 1)

        # Top-down path
        self.up_conv1 = ConvBNSiLU(c5, c4, 1)
        self.td_c2f1 = C2f(c4 + c4, c4, n=n, shortcut=False)
        self.up_conv2 = ConvBNSiLU(c4, c3, 1)
        self.td_c2f2 = C2f(c3 + c3, c3, n=n, shortcut=False)

        # Bottom-up path
        self.down_conv1 = ConvBNSiLU(c3, c3, 3, 2)
        self.bu_c2f1 = C2f(c3 + c4, c4, n=n, shortcut=False)
        self.down_conv2 = ConvBNSiLU(c4, c4, 3, 2)
        self.bu_c2f2 = C2f(c4 + c5, c5, n=n, shortcut=False)

        self.out_channels = [c3, c4, c5]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # Top-down
        up5 = self.up_conv1(p5)
        td4 = self.td_c2f1(torch.cat([F.interpolate(up5, size=p4.shape[2:], mode="nearest"), p4], 1))
        up4 = self.up_conv2(td4)
        td3 = self.td_c2f2(torch.cat([F.interpolate(up4, size=p3.shape[2:], mode="nearest"), p3], 1))

        # Bottom-up
        bu4 = self.bu_c2f1(torch.cat([self.down_conv1(td3), td4], 1))
        bu5 = self.bu_c2f2(torch.cat([self.down_conv2(bu4), p5], 1))

        return [td3, bu4, bu5]
