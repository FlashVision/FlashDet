"""YOLOX PAFPN neck.

Independent implementation based on:
    Ge et al., "YOLOX: Exceeding YOLO Series in 2021",
    arXiv:2107.08430, 2021.

License: MIT (same as FlashDet)
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.models.layers.yolo_blocks import ConvBNSiLU
from flashdet.models.backbone.yolox_backbone import CSPBlock


class YOLOXNeck(nn.Module):
    """PAFPN neck for YOLOX with CSP blocks.

    Args:
        in_channels: Channel sizes from backbone [P3, P4, P5].
        depth_mult: Depth multiplier for CSP blocks.
    """

    def __init__(self, in_channels: List[int], depth_mult: float = 1.0):
        super().__init__()
        c3, c4, c5 = in_channels
        n = max(round(3 * depth_mult), 1)

        # Top-down (reduce + upsample + concat + CSP)
        self.reduce1 = ConvBNSiLU(c5, c4, 1)
        self.csp_td1 = CSPBlock(c4 * 2, c4, n=n, shortcut=False)
        self.reduce2 = ConvBNSiLU(c4, c3, 1)
        self.csp_td2 = CSPBlock(c3 * 2, c3, n=n, shortcut=False)

        # Bottom-up (downsample + concat + CSP)
        self.down1 = ConvBNSiLU(c3, c3, 3, 2)
        self.csp_bu1 = CSPBlock(c3 + c4, c4, n=n, shortcut=False)
        self.down2 = ConvBNSiLU(c4, c4, 3, 2)
        self.csp_bu2 = CSPBlock(c4 + c5, c5, n=n, shortcut=False)

        self.out_channels = [c3, c4, c5]

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features

        # Top-down
        td5 = self.reduce1(p5)
        td4 = self.csp_td1(torch.cat([F.interpolate(td5, size=p4.shape[2:], mode="nearest"), p4], 1))
        td4_r = self.reduce2(td4)
        td3 = self.csp_td2(torch.cat([F.interpolate(td4_r, size=p3.shape[2:], mode="nearest"), p3], 1))

        # Bottom-up
        bu4 = self.csp_bu1(torch.cat([self.down1(td3), td4], 1))
        bu5 = self.csp_bu2(torch.cat([self.down2(bu4), p5], 1))

        return [td3, bu4, bu5]
