from .nanodet_head import FlashDetHead, Integral, DepthwiseConvModule
from .aux_head import SimpleConvHead
from .obb_head import OBBHead
from .yolo_head import YOLODetectionHead, DualHeadOne2One, DualHeadOne2Many, PGIAuxBranch
from .detr_head import DETRHead
from .rt_detr_decoder import RTDETRDecoder
from .grounding_dino_decoder import GroundingDINODecoder
from .e2e_head import E2EDetHead, E2EDualHead
from flashdet.registry import HEADS

HEADS.register("FlashDetHead")(FlashDetHead)
HEADS.register("SimpleConvHead")(SimpleConvHead)
HEADS.register("YOLODetectionHead")(YOLODetectionHead)
HEADS.register("DETRHead")(DETRHead)
HEADS.register("RTDETRDecoder")(RTDETRDecoder)
HEADS.register("GroundingDINODecoder")(GroundingDINODecoder)
HEADS.register("E2EDualHead")(E2EDualHead)

__all__ = [
    "FlashDetHead",
    "Integral",
    "DepthwiseConvModule",
    "SimpleConvHead",
    "OBBHead",
    "YOLODetectionHead",
    "DualHeadOne2One",
    "DualHeadOne2Many",
    "PGIAuxBranch",
    "DETRHead",
    "RTDETRDecoder",
    "GroundingDINODecoder",
    "E2EDetHead",
    "E2EDualHead",
]
