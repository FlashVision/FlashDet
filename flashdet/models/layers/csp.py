"""CSP-style blocks: C2f, C3k2, CSPRepLayer, ELAN, GELAN."""

import torch
import torch.nn as nn

from .conv import ConvBNSiLU
from .bottleneck import Bottleneck
from .repvgg import RepVGGBlock


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions — standard YOLO building block."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(self.c, self.c, 3),
                ConvBNSiLU(self.c, self.c, 3),
            )
            for _ in range(n)
        ])
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.blocks:
            out = blk(y[-1])
            if self.shortcut and out.shape == y[-1].shape:
                out = out + y[-1]
            y.append(out)
        return self.cv2(torch.cat(y, 1))


class C3k2(nn.Module):
    """C3k2 block — CSP bottleneck with 2 convolutions and flexible kernels.

    A variant of C2f that uses Bottleneck blocks with configurable kernel
    sizes (defaulting to 3x3), enabling richer multi-scale feature extraction.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        k: int = 3,
    ):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch)
        self.blocks = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut=shortcut, k=(k, k))
            for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.blocks:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, 1))


class CSPRepLayer(nn.Module):
    """CSP bottleneck layer with RepVGG blocks used in the CCFM."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3, expansion: float = 1.0):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[RepVGGBlock(hidden, hidden) for _ in range(num_blocks)]
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(torch.cat([self.blocks(self.conv1(x)), self.conv2(x)], dim=1))


class ELAN(nn.Module):
    """Efficient Layer Aggregation Network block.

    Splits input into two branches: one passes through a series of
    convolutions with intermediate concatenations for gradient-rich
    aggregation.
    """

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, num_blocks: int = 2):
        super().__init__()
        mid_ch = mid_ch or out_ch // 2
        self.conv1 = ConvBNSiLU(in_ch, mid_ch)
        self.conv2 = ConvBNSiLU(in_ch, mid_ch)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                ConvBNSiLU(mid_ch, mid_ch, 3),
                ConvBNSiLU(mid_ch, mid_ch, 3),
            ))

        self.merge = ConvBNSiLU(mid_ch * (2 + num_blocks), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        parts = [x1, x2]
        cur = x2
        for blk in self.blocks:
            cur = blk(cur)
            parts.append(cur)
        return self.merge(torch.cat(parts, dim=1))


class GELAN(nn.Module):
    """Generalized ELAN — YOLOv9 backbone based on ELAN with flexible routing."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = None, num_blocks: int = 3):
        super().__init__()
        mid_ch = mid_ch or out_ch // 2
        self.conv1 = ConvBNSiLU(in_ch, mid_ch)
        self.conv2 = ConvBNSiLU(in_ch, mid_ch)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                ConvBNSiLU(mid_ch, mid_ch, 3),
                ConvBNSiLU(mid_ch, mid_ch, 3),
            ))

        total_concat = mid_ch * (2 + num_blocks)
        self.transition = ConvBNSiLU(total_concat, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        outs = [x1, x2]
        cur = x2
        for blk in self.blocks:
            cur = blk(cur)
            outs.append(cur)
        return self.transition(torch.cat(outs, dim=1))
