from .ghost_pan import GhostPAN, GhostBottleneck, GhostModule, GhostBlocks
from flashdet.models.layers.conv_module import ConvModule, DepthwiseConvModule
from .hybrid_encoder import HybridEncoder, AIFI
from .yolov9_neck import YOLOv9Neck
from .yolov10_neck import YOLOv10Neck
from .yolov11_neck import YOLOv11Neck
from flashdet.registry import NECKS

NECKS.register("GhostPAN")(GhostPAN)
NECKS.register("HybridEncoder")(HybridEncoder)
NECKS.register("YOLOv9Neck")(YOLOv9Neck)
NECKS.register("YOLOv10Neck")(YOLOv10Neck)
NECKS.register("YOLOv11Neck")(YOLOv11Neck)

__all__ = [
    "GhostPAN",
    "GhostBottleneck",
    "GhostModule",
    "GhostBlocks",
    "ConvModule",
    "DepthwiseConvModule",
    "HybridEncoder",
    "AIFI",
    "YOLOv9Neck",
    "YOLOv10Neck",
    "YOLOv11Neck",
]
