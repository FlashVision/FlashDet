"""MicroBlock — Unified building block for FlashDet-Micro.

A single reparameterizable block used identically across backbone, neck,
and head for a consistent, ultra-efficient architecture.

Key innovations:
  - Dual DW conv (3x3 || 5x5) fuses to single 5x5 at inference
  - Efficient Channel Attention (ECA) for ~0-cost accuracy boost
  - Residual connection for stable gradient flow

Reference:
  ECA — Wang et al., "ECA-Net: Efficient Channel Attention", CVPR 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECALayer(nn.Module):
    """Efficient Channel Attention — global pool → 1D conv → sigmoid.

    Adds channel-wise recalibration with only k learnable parameters.
    """

    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                       # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)         # [B, 1, C]
        y = self.conv(y)                             # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)        # [B, C, 1, 1]
        return x * y.sigmoid()


class MicroBlock(nn.Module):
    """Unified residual block with reparameterizable DW conv + ECA.

    Training:  x → [DW 3x3 + DW 5x5] → BN → PW 1x1 → BN → Act → ECA → (+x)
    Inference: x → DW 5x5 (fused) → BN → PW 1x1 → BN → Act → ECA → (+x)

    The dual DW branches capture multi-scale features during training,
    then fuse into a single 5x5 DW conv at deployment (zero extra cost).
    """

    def __init__(self, channels: int, activation: str = "LeakyReLU"):
        super().__init__()
        self.dw5x5 = nn.Conv2d(
            channels, channels, 5, 1, 2, groups=channels, bias=False,
        )
        self.dw3x3 = nn.Conv2d(
            channels, channels, 3, 1, 1, groups=channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        self.eca = ECALayer(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "dw3x3"):
            out = self.dw5x5(x) + self.dw3x3(x)
        else:
            out = self.dw5x5(x)
        out = self.bn1(out)
        out = self.pw(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.eca(out)
        return out + x

    @torch.no_grad()
    def fuse(self):
        """Fuse 3x3 branch into 5x5 for deployment."""
        if not hasattr(self, "dw3x3"):
            return
        w3 = F.pad(self.dw3x3.weight, [1, 1, 1, 1])
        self.dw5x5.weight.data.add_(w3)
        del self.dw3x3


class MicroDown(nn.Module):
    """Stride-2 spatial reduction with channel expansion.

    DW Conv 3x3 stride-2 → BN → Act → PW Conv 1x1 → BN → Act
    """

    def __init__(
        self, in_channels: int, out_channels: int, activation: str = "LeakyReLU",
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels, in_channels, 3, 2, 1, groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "SiLU":
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x
