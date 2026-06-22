"""YOLOv10 efficient PANet neck."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.models.layers import ConvBNSiLU, SCDown, C2f


class YOLOv10Neck(nn.Module):
    """Efficient PANet neck for YOLOv10."""

    def __init__(self, in_channels: List[int], out_channels: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvBNSiLU(ch, out_channels) for ch in in_channels])
        self.td_blocks = nn.ModuleList()
        self.bu_downs = nn.ModuleList()
        self.bu_blocks = nn.ModuleList()

        for _ in range(len(in_channels) - 1):
            self.td_blocks.append(C2f(out_channels * 2, out_channels, n=2))
            self.bu_downs.append(SCDown(out_channels, out_channels))
            self.bu_blocks.append(C2f(out_channels * 2, out_channels, n=2))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        lats = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(lats) - 1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i - 1].shape[2:], mode="nearest")
            lats[i - 1] = self.td_blocks[i - 1](torch.cat([lats[i - 1], up], dim=1))

        for i in range(len(lats) - 1):
            down = self.bu_downs[i](lats[i])
            lats[i + 1] = self.bu_blocks[i](torch.cat([down, lats[i + 1]], dim=1))

        return lats
