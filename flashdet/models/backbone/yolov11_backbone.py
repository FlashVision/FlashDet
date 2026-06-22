"""YOLOv11 backbone — C3k2 with SPPF and optional C2PSA."""

from typing import List

import torch
import torch.nn as nn

from flashdet.models.layers import ConvBNSiLU, C3k2, SPPF, C2PSA


class YOLOv11Backbone(nn.Module):
    """C3k2-based backbone for YOLOv11 with SPPF and optional C2PSA.

    Args:
        width_mult: Channel width multiplier (0.25=n, 0.5=s, 1.0=m, 1.25=l, 1.5=x).
        depth_mult: Block depth multiplier.
        use_c2psa: Use C2PSA attention in the last backbone stage.
    """

    def __init__(self, width_mult: float = 1.0, depth_mult: float = 1.0, use_c2psa: bool = True):
        super().__init__()
        base = [64, 128, 256, 512]
        channels = [max(int(c * width_mult), 16) for c in base]
        n_blocks = [max(int(3 * depth_mult), 1) for _ in range(4)]

        self.stem = ConvBNSiLU(3, channels[0], 3, 2)
        self.stage1 = nn.Sequential(
            ConvBNSiLU(channels[0], channels[1], 3, 2),
            C3k2(channels[1], channels[1], n=n_blocks[0]),
        )
        self.stage2 = nn.Sequential(
            ConvBNSiLU(channels[1], channels[2], 3, 2),
            C3k2(channels[2], channels[2], n=n_blocks[1]),
        )

        stage3_layers: List[nn.Module] = [
            ConvBNSiLU(channels[2], channels[3], 3, 2),
            C3k2(channels[3], channels[3], n=n_blocks[2]),
            SPPF(channels[3], channels[3]),
        ]
        if use_c2psa:
            stage3_layers.append(C2PSA(channels[3], channels[3], n=1))
        self.stage3 = nn.Sequential(*stage3_layers)

        self.out_channels = [channels[1], channels[2], channels[3]]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]
