from .nanodet_head import FlashDetHead, Integral, DepthwiseConvModule
from .aux_head import SimpleConvHead
from .obb_head import OBBHead
from flashdet.registry import HEADS

HEADS.register("FlashDetHead")(FlashDetHead)
HEADS.register("SimpleConvHead")(SimpleConvHead)

__all__ = [
    "FlashDetHead",
    "Integral",
    "DepthwiseConvModule",
    "SimpleConvHead",
    "OBBHead",
]
