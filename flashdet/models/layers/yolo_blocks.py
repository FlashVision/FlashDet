"""YOLO-family layer primitives (C2f, GELAN, SCDown, PSA, etc.).

Independent implementation of standard building blocks described in:
  - Wang et al., "YOLOv9: Learning What You Want to Learn Using
    Programmable Gradient Information", arXiv:2402.13616, 2024.
  - Wang et al., "YOLOv10: Real-Time End-to-End Object Detection",
    arXiv:2405.14458, 2024. (Original code: Apache-2.0, THU-MIG/yolov10)
  - Jocher et al., YOLO11, Ultralytics, 2024.

These implementations are written from scratch based on the architectural
descriptions in the above papers. No code was copied from AGPL-licensed sources.

License: MIT (same as FlashDet)
"""

import torch
import torch.nn as nn
from typing import List


class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU — standard YOLO building block."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = None, g: int = 1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck (1x1 reduce + 3x3 expand) with residual."""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        mid = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, mid, 1)
        self.cv2 = ConvBNSiLU(mid, out_ch, 3)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8/v10/v11 style).

    Splits channels, applies N bottleneck blocks on one half, then
    concatenates and projects. More efficient than C3 due to split design.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch, 1)
        self.blocks = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut=shortcut) for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.blocks:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SCDown(nn.Module):
    """Spatial-Channel Decoupled Downsampling (YOLOv10).

    Reduces spatial dimensions and adjusts channels in separate steps.
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 2):
        super().__init__()
        self.pointwise = ConvBNSiLU(in_ch, out_ch, 1)
        self.spatial = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, k, s, k // 2, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.pointwise(x))


class PSA(nn.Module):
    """Partial Self-Attention module (YOLOv10).

    Applies multi-head self-attention on a portion of the channels
    while keeping the rest unchanged, balancing efficiency and accuracy.
    """

    def __init__(self, channels: int, num_heads: int = 4, attn_ratio: float = 0.5):
        super().__init__()
        self.attn_ch = int(channels * attn_ratio)
        self.pass_ch = channels - self.attn_ch

        self.norm = nn.LayerNorm(self.attn_ch)
        self.attn = nn.MultiheadAttention(self.attn_ch, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.attn_ch, self.attn_ch * 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.attn_ch * 2, self.attn_ch),
        )
        self.proj = ConvBNSiLU(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_attn, x_pass = x.split([self.attn_ch, self.pass_ch], dim=1)

        tokens = x_attn.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = tokens + attn_out
        attn_out = attn_out + self.ffn(attn_out)
        x_attn = attn_out.transpose(1, 2).reshape(B, self.attn_ch, H, W)

        out = torch.cat([x_attn, x_pass], dim=1)
        return self.proj(out)


class DownSample(nn.Module):
    """Stride-2 downsampling for YOLOv9 (Conv 3x3 stride 2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNSiLU(in_ch, out_ch, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GELAN(nn.Module):
    """Generalized Efficient Layer Aggregation Network block (YOLOv9).

    Multi-branch aggregation with configurable depth, enabling
    efficient gradient flow across branches.
    """

    def __init__(self, in_ch: int, out_ch: int, num_blocks: int = 3, e: float = 0.5):
        super().__init__()
        mid = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * mid, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(mid, mid, 3),
                ConvBNSiLU(mid, mid, 3),
            ) for _ in range(num_blocks)
        ])
        self.cv2 = ConvBNSiLU((2 + num_blocks) * mid, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for blk in self.blocks:
            y.append(blk(y[-1]))
        return self.cv2(torch.cat(y, 1))


class RepConv(nn.Module):
    """Re-parameterizable convolution (3x3 + 1x1 branches, fused at inference)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_ch, out_ch, k, s)
        self.conv2 = ConvBNSiLU(in_ch, out_ch, 1, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.conv2(x)
