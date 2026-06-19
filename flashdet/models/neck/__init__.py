from .ghost_pan import GhostPAN, GhostBottleneck, GhostModule, GhostBlocks
from .conv_module import ConvModule, DepthwiseConvModule
from flashdet.registry import NECKS

NECKS.register("GhostPAN")(GhostPAN)

__all__ = [
    "GhostPAN",
    "GhostBottleneck", 
    "GhostModule",
    "GhostBlocks",
    "ConvModule",
    "DepthwiseConvModule"
]
