"""Standard bottleneck block with optional shortcut."""

from typing import Tuple

import torch
import torch.nn as nn

from .conv import ConvBNSiLU


class Bottleneck(nn.Module):
    """Standard bottleneck with optional shortcut."""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, k: Tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        mid = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, mid, k[0])
        self.cv2 = ConvBNSiLU(mid, out_ch, k[1])
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out
