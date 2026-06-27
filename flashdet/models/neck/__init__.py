from .pico_neck import PicoNeck, LiteBlock, LiteModule, LiteBlocks
from .yolov8_neck import YOLOv8Neck
from .yolov9_neck import YOLOv9Neck
from .yolov10_neck import YOLOv10Neck
from .yolov11_neck import YOLOv11Neck
from .yolox_neck import YOLOXNeck
from flashdet.registry import NECKS

NECKS.register("PicoNeck")(PicoNeck)
NECKS.register("YOLOv8Neck")(YOLOv8Neck)
NECKS.register("YOLOv9Neck")(YOLOv9Neck)
NECKS.register("YOLOv10Neck")(YOLOv10Neck)
NECKS.register("YOLOv11Neck")(YOLOv11Neck)
NECKS.register("YOLOXNeck")(YOLOXNeck)

__all__ = [
    "PicoNeck",
    "LiteBlock",
    "LiteModule",
    "LiteBlocks",
    "YOLOv8Neck",
    "YOLOv9Neck",
    "YOLOv10Neck",
    "YOLOv11Neck",
    "YOLOXNeck",
]
