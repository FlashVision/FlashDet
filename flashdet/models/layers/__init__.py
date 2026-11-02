"""Shared building blocks used across multiple architectures."""

from .conv import ConvBlock
from .conv_module import ConvModule, DepthwiseConvModule
from .pooling import SpatialPool
from .reparam import (
    AttentionBlock,
    FusedDWConv,
    GhostBlock,
    LCNetBlock,
    MultiScaleConv,
    PICO_BLOCK_FAMILIES,
    PicoBlock,
    RepBlock,
    SEBlock,
    SSMBlock,
    StrideDown,
    make_pico_block,
)
from .yolo_blocks import (
    ConvBNSiLU, Bottleneck, C2f, SCDown, PSA, DownSample, GELAN, RepConv,
)

__all__ = [
    "ConvBlock",
    "ConvModule",
    "DepthwiseConvModule",
    "SpatialPool",
    "PicoBlock",
    "StrideDown",
    "MultiScaleConv",
    "FusedDWConv",
    # Pico-Zoo family blocks
    "SEBlock",
    "RepBlock",
    "GhostBlock",
    "AttentionBlock",
    "SSMBlock",
    "LCNetBlock",
    "PICO_BLOCK_FAMILIES",
    "make_pico_block",
    # YOLO-family blocks
    "ConvBNSiLU",
    "Bottleneck",
    "C2f",
    "SCDown",
    "PSA",
    "DownSample",
    "GELAN",
    "RepConv",
]
