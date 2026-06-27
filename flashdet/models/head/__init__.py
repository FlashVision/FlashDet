from .e2e_head import E2EDetHead, E2EDualHead
from .yolo_head import YOLODetectionHead, DualHeadOne2One, DualHeadOne2Many, PGIAuxBranch
from .yolox_head import YOLOXHead
from flashdet.registry import HEADS

HEADS.register("E2EDetHead")(E2EDetHead)
HEADS.register("E2EDualHead")(E2EDualHead)
HEADS.register("YOLODetectionHead")(YOLODetectionHead)
HEADS.register("DualHeadOne2One")(DualHeadOne2One)
HEADS.register("DualHeadOne2Many")(DualHeadOne2Many)
HEADS.register("PGIAuxBranch")(PGIAuxBranch)
HEADS.register("YOLOXHead")(YOLOXHead)

__all__ = [
    "E2EDetHead",
    "E2EDualHead",
    "YOLODetectionHead",
    "DualHeadOne2One",
    "DualHeadOne2Many",
    "PGIAuxBranch",
    "YOLOXHead",
]
