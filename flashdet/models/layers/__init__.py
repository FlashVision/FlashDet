"""Shared building blocks used across multiple architectures."""

from .conv import ConvBlock
from .conv_module import ConvModule, DepthwiseConvModule
from .pooling import SpatialPool
from .reparam import PicoBlock, StrideDown, MultiScaleConv, FusedDWConv
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
