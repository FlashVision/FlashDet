"""Partial Self-Attention modules."""

import torch
import torch.nn as nn

from .conv import ConvBNSiLU
from .bottleneck import Bottleneck


class PSA(nn.Module):
    """Partial Self-Attention module for efficient global context."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)

        return x + self.norm(self.proj(out))


class C2PSA(nn.Module):
    """C2f with Partial Self-Attention for YOLOv11.

    Integrates lightweight self-attention into the C2f architecture
    for improved global context modelling.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.c)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_ch)

        self.attn_blocks = nn.ModuleList()
        for _ in range(n):
            self.attn_blocks.append(nn.Sequential(
                ConvBNSiLU(self.c, self.c, 3),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.c, self.c),
                nn.SiLU(inplace=True),
                nn.Linear(self.c, self.c),
                nn.Sigmoid(),
            ))
        self.bottlenecks = nn.ModuleList([
            Bottleneck(self.c, self.c) for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for attn, bn in zip(self.attn_blocks, self.bottlenecks):
            feat = bn(y[-1])
            B, C, H, W = feat.shape
            scale = attn(feat).view(B, C, 1, 1)
            y.append(feat * scale)
        return self.cv2(torch.cat(y, 1))
