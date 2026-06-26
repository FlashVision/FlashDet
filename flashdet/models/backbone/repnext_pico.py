"""
RepNeXt-Pico Backbone for FlashDet.

Lightweight backbone inspired by RepNeXt (https://arxiv.org/abs/2406.16004):
  - Multi-scale ChunkConv: 4-way channel split with identity, 3×3, 5×5, 7×7
    depthwise convolutions for diverse spatial feature extraction.
  - CopyConv-style downsampling: two parallel DW paths (3×3 ‖ 5×5) concat
    to double channels at each stride-2 transition.
  - Structural reparameterization: multi-branch training paths fuse into
    single depthwise convolutions at inference (zero extra cost).

Drop-in replacement for ShuffleNetV2-0.5x:
  Output channels [48, 96, 192] at strides [8, 16, 32].
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reparameterizable primitives
# ---------------------------------------------------------------------------

class ConvBN(nn.Sequential):
    """Conv2d + BatchNorm2d with BN-folding support."""

    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=bias),
            nn.BatchNorm2d(out_ch),
        )

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        conv, bn = self[0], self[1]
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - w * bn.running_mean
        if conv.bias is not None:
            b = b + w * conv.bias
        w = w[:, None, None, None] * conv.weight
        m = nn.Conv2d(
            w.size(1) * conv.groups, w.size(0), w.shape[2:],
            stride=conv.stride, padding=conv.padding,
            dilation=conv.dilation, groups=conv.groups,
            device=conv.weight.device,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepDWConvS(nn.Module):
    """Reparameterizable small DW conv (from RepNeXt).

    Training:  3×3 + (1×3) + (3×1) + dilated-2×2  (4 parallel branches)
    Inference: single 3×3 DW conv  (via structural reparameterization)
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


# ---------------------------------------------------------------------------
# Core blocks
# ---------------------------------------------------------------------------

class PicoChunkConv(nn.Module):
    """Multi-scale depthwise convolution via channel chunking.

    Splits input channels into 4 equal groups and applies:
      0) Identity  – free skip, gradient highway
      1) RepDW 3×3 – local features (fuses to single 3×3)
      2) DW 5×5    – medium receptive field
      3) DW 7×7    – large receptive field via (1×7)·(7×1) decomposition

    At inference, all branches collapse to simple DW convolutions.
    """

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, f"channels ({channels}) must be divisible by 4"
        c = channels // 4
        self._c = c
        self.dw_s = RepDWConvS(c)
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


class RepNeXtPicoBlock(nn.Module):
    """RepNeXt-style block sized for Pico.

    Architecture:  x → ChunkConv → BN → PW(1×1) → BN → Act → (+x)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.chunk_conv = PicoChunkConv(dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.bn2(self.pw(self.bn1(self.chunk_conv(x)))))


class PicoCopyDown(nn.Module):
    """Stride-2 downsample that doubles channels via parallel DW paths.

    Inspired by RepNeXt CopyConv: two depthwise convolutions with
    different kernel sizes (3×3 ‖ 5×5) → concatenate → BN → Act → PW → BN
    → Act + residual.  This captures multi-scale information during the
    spatial reduction step.
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
# Full backbone
# ---------------------------------------------------------------------------

class RepNeXtPico(nn.Module):
    """RepNeXt-Pico backbone — drop-in replacement for ShuffleNetV2-0.5x.

    ┌─────────┐
    │  Stem   │  Conv3×3 stride-2 → BN → Act
    │ MaxPool │  stride-2
    ├─────────┤  stride 4, 24 ch
    │ Stage 0 │  CopyDown(24→48) + 2× PicoBlock(48)
    ├─────────┤  stride 8, 48 ch
    │ Stage 1 │  CopyDown(48→96) + 3× PicoBlock(96)
    ├─────────┤  stride 16, 96 ch
    │ Stage 2 │  CopyDown(96→192) + 1× PicoBlock(192)
    └─────────┘  stride 32, 192 ch

    Outputs [48, 96, 192] at strides [8, 16, 32] — identical interface
    to ShuffleNetV2(model_size="0.5x", out_stages=(2, 3, 4)).

    Parameter budget: ~135K (comparable to ShuffleNetV2-0.5x backbone).
    """

    def __init__(
        self,
        stem_channels: int = 24,
        stage_channels: Tuple[int, ...] = (48, 96, 192),
        stage_depths: Tuple[int, ...] = (2, 3, 1),
        out_stages: Tuple[int, ...] = (0, 1, 2),
        activation: str = "LeakyReLU",
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.out_stages = out_stages
        self.out_channels = [stage_channels[s] for s in out_stages]

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.MaxPool2d(3, 2, 1)

        in_ch = stem_channels
        self.stages = nn.ModuleList()
        for ch, depth in zip(stage_channels, stage_depths):
            layers: List[nn.Module] = [PicoCopyDown(in_ch)]
            for _ in range(depth):
                layers.append(RepNeXtPicoBlock(ch))
            self.stages.append(nn.Sequential(*layers))
            in_ch = ch

        self._init_weights()
        self.pretrained_loaded = False

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.pool(self.stem(x))
        outputs: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_stages:
                outputs.append(x)
        return outputs

    @torch.no_grad()
    def fuse(self):
        """Fuse all reparameterizable branches for deployment."""
        def _fuse_recursive(module):
            for name, child in module.named_children():
                if hasattr(child, "fuse") and child is not self:
                    fused = child.fuse()
                    if fused is not None:
                        setattr(module, name, fused)
                        _fuse_recursive(fused)
                    else:
                        _fuse_recursive(child)
                else:
                    _fuse_recursive(child)
        _fuse_recursive(self)
        logger.info("RepNeXtPico: fused all reparameterizable branches")
        return self
