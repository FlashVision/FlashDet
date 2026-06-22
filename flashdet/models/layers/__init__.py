"""Shared building blocks used across multiple architectures.

To add a new layer or block:
  1. Create a new file in this directory (e.g. ``my_block.py``).
  2. Import and export it here.
  3. Use it from any backbone, neck, or head via
     ``from flashdet.models.layers import MyBlock``.
"""

from .conv import ConvBNSiLU, DownSample
from .conv_module import ConvModule, DepthwiseConvModule
from .repvgg import RepConv, RepVGGBlock
from .csp import C2f, C3k2, CSPRepLayer, ELAN, GELAN
from .bottleneck import Bottleneck
from .sppf import SPPF
from .psa import PSA, C2PSA
from .scdown import SCDown

__all__ = [
    # Basic conv blocks
    "ConvBNSiLU",
    "DownSample",
    "ConvModule",
    "DepthwiseConvModule",
    # Re-param blocks
    "RepConv",
    "RepVGGBlock",
    # CSP / aggregation blocks
    "C2f",
    "C3k2",
    "CSPRepLayer",
    "ELAN",
    "GELAN",
    # Bottleneck
    "Bottleneck",
    # Pooling
    "SPPF",
    # Attention
    "PSA",
    "C2PSA",
    # Downsampling
    "SCDown",
]
