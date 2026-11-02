"""Reparameterizable building blocks for FlashDet backbones.

Multi-scale depthwise convolutions and stride-2 downsample modules
with structural reparameterization — multi-branch training collapses
to single convolutions at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedDWConv(nn.Module):
    """Reparameterizable small depthwise convolution.

    Training:  3x3 + (1x3) + (3x1) + dilated-2x2  (4 parallel branches)
    Inference: single 3x3 DW conv  (via structural reparameterization)
    """

    def __init__(self, channels: int):
        super().__init__()
        kw = dict(in_channels=channels, out_channels=channels,
                  groups=channels, bias=False)
        self.conv_3x3 = nn.Conv2d(kernel_size=3, padding=1, **kw)
        self.conv_1x3 = nn.Conv2d(kernel_size=(1, 3), padding=(0, 1), **kw)
        self.conv_3x1 = nn.Conv2d(kernel_size=(3, 1), padding=(1, 0), **kw)
        self.conv_d22 = nn.Conv2d(kernel_size=2, dilation=2, padding=1, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv_1x3"):
            return (self.conv_3x3(x) + self.conv_1x3(x)
                    + self.conv_3x1(x) + self.conv_d22(x))
        return self.conv_3x3(x)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        if not hasattr(self, "conv_1x3"):
            return self.conv_3x3
        w = self.conv_3x3.weight.clone()
        w += F.pad(self.conv_1x3.weight, [0, 0, 1, 1])
        w += F.pad(self.conv_3x1.weight, [1, 1, 0, 0])
        d22_w = F.conv_transpose2d(
            self.conv_d22.weight,
            torch.ones(1, 1, 1, 1, device=w.device),
            stride=2,
        )
        w += d22_w
        self.conv_3x3.weight.data.copy_(w)
        del self.conv_1x3, self.conv_3x1, self.conv_d22
        return self.conv_3x3


class MultiScaleConv(nn.Module):
    """Multi-scale depthwise convolution via channel chunking.

    Splits input channels into 4 equal groups and applies:
      0) Identity  -- free skip, gradient highway
      1) Fused DW 3x3 -- local features (fuses to single 3x3)
      2) DW 5x5    -- medium receptive field
      3) DW 7x7    -- large receptive field via (1x7)*(7x1) decomposition

    At inference, all branches collapse to simple DW convolutions.
    """

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, f"channels ({channels}) must be divisible by 4"
        c = channels // 4
        self._c = c
        self.dw_s = FusedDWConv(c)
        self.dw_m = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.dw_l_h = nn.Conv2d(c, c, (1, 7), 1, (0, 3), groups=c, bias=False)
        self.dw_l_v = nn.Conv2d(c, c, (7, 1), 1, (3, 0), groups=c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i, s, m, l = x.chunk(4, dim=1)
        if hasattr(self, "dw_l_h"):
            l_out = self.dw_l_v(self.dw_l_h(l))
        else:
            l_out = self.dw_l(l)
        return torch.cat([i, self.dw_s(s), self.dw_m(m), l_out], dim=1)

    @torch.no_grad()
    def fuse(self):
        self.dw_s.fuse()
        if not hasattr(self, "dw_l_h"):
            return
        c = self._c
        w_7x7 = torch.einsum(
            "bcnx,bcyn->bcyx",
            self.dw_l_h.weight, self.dw_l_v.weight,
        )
        fused = nn.Conv2d(
            c, c, 7, 1, 3, groups=c, bias=False,
            device=w_7x7.device,
        )
        fused.weight.data.copy_(w_7x7)
        self.dw_l = fused
        del self.dw_l_h, self.dw_l_v


class PicoBlock(nn.Module):
    """Residual block with multi-scale depthwise convolution.

    Architecture:  x -> MultiScaleConv -> BN -> PW(1x1) -> BN -> Act -> (+x)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.chunk_conv = MultiScaleConv(dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.bn2(self.pw(self.bn1(self.chunk_conv(x)))))


class StrideDown(nn.Module):
    """Stride-2 downsample that doubles channels via parallel DW paths.

    Two depthwise convolutions with different kernel sizes (3x3 || 5x5)
    are concatenated then projected, capturing multi-scale information
    during the spatial reduction step.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        out = in_channels * 2
        self.dw_s = nn.Conv2d(
            in_channels, in_channels, 3, 2, 1,
            groups=in_channels, bias=False,
        )
        self.dw_l = nn.Conv2d(
            in_channels, in_channels, 5, 2, 2,
            groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out)
        self.pw = nn.Conv2d(out, out, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(torch.cat([self.dw_s(x), self.dw_l(x)], dim=1)))
        return x + self.act(self.bn2(self.pw(x)))


# ---------------------------------------------------------------------------
# Pico-Zoo blocks — one lightweight block per architecture family.
#
# Every block is channel-preserving and can be used in both the backbone and
# the neck.  Only the Rep family is structurally reparameterizable; the other
# families are cheap, deployment-friendly CNN blocks inspired by the listed
# architectures.
# ---------------------------------------------------------------------------

def _make_act(activation: str = "LeakyReLU") -> nn.Module:
    if activation is None:
        return nn.Identity()
    if activation.lower() in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(0.1, inplace=True)
    return nn.SiLU(inplace=True)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention (very cheap at low width)."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.act(self.fc1(self.pool(x)))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class RepBlock(nn.Module):
    """Category A — reparameterized depthwise block.

    Training uses 3x3 + 1x3 + 3x1 + dilated-2x2 depthwise branches.
    ``fuse()`` collapses them into a single 3x3 depthwise convolution.
    """

    def __init__(self, channels: int, activation: str = "LeakyReLU"):
        super().__init__()
        self.dw = FusedDWConv(channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.bn2(self.pw(self.bn1(self.dw(x)))))

    @torch.no_grad()
    def fuse(self):
        self.dw = self.dw.fuse()
        return self


class GhostBlock(nn.Module):
    """Category C — cheap feature generation (GhostNetV2/MobileNetV4 style)."""

    def __init__(self, channels: int, activation: str = "LeakyReLU"):
        super().__init__()
        mid = max(1, channels // 2)
        new = max(1, channels - mid)
        self.primary = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            _make_act(activation),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(mid, new, 3, 1, 1, groups=mid, bias=False),
            nn.BatchNorm2d(new),
            _make_act(activation),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid + new, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        y = torch.cat([y, self.cheap(y)], dim=1)
        return x + self.act(self.project(y))


class AttentionBlock(nn.Module):
    """Category B/G — local depthwise mixing + channel/spatial gating."""

    def __init__(self, channels: int, expansion: int = 2,
                 activation: str = "LeakyReLU"):
        super().__init__()
        hidden = max(channels, channels * expansion)
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.dw(x)))
        y = self.se(y)
        y = self.act(self.bn2(self.pw1(y)))
        y = self.bn3(self.pw2(y))
        return x + self.act(y)


class SSMBlock(nn.Module):
    """Category D/H — MambaOut-style gated token mixer.

    Uses a gated depthwise convolution instead of selective-scan kernels so it
    runs on CPU/edge devices with no CUDA-only Mamba dependency.
    """

    def __init__(self, channels: int, activation: str = "LeakyReLU"):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.gate = nn.Conv2d(channels, channels, 1, bias=True)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        gate = torch.sigmoid(self.gate(y))
        y = self.act(self.dw(y)) * gate
        return x + self.act(self.bn(self.pw(y)))


class LCNetBlock(nn.Module):
    """Category E/F — PP-LCNet / ShuffleNet-style depthwise + SE block."""

    def __init__(self, channels: int, kernel_size: int = 5,
                 activation: str = "LeakyReLU"):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size, 1, kernel_size // 2,
            groups=channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = _make_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.se(self.act(self.bn1(self.dw(x))))
        return x + self.act(self.bn2(self.pw(y)))


PICO_BLOCK_FAMILIES = ("rep", "ghost", "attention", "mamba", "lcnet")


def make_pico_block(family: str, channels: int,
                    activation: str = "LeakyReLU") -> nn.Module:
    """Build one channel-preserving Pico-Zoo block by family name."""
    family = (family or "rep").lower()
    if family in ("rep", "reparam", "repvit", "mobileone", "fastvit"):
        return RepBlock(channels, activation)
    if family in ("ghost", "ghostnet", "efficientnet", "fasternet"):
        return GhostBlock(channels, activation)
    if family in ("attention", "attn", "transformer", "hybrid"):
        return AttentionBlock(channels, activation=activation)
    if family in ("mamba", "ssm", "mambaout"):
        return SSMBlock(channels, activation)
    if family in ("lcnet", "lc", "shufflenet", "esnet", "peleenet"):
        return LCNetBlock(channels, activation=activation)
    raise ValueError(
        f"Unknown Pico block family '{family}'. "
        f"Choose from: {', '.join(PICO_BLOCK_FAMILIES)}"
    )
