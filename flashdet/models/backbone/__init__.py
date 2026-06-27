from .lite_backbone import LiteBackbone, LiteUnit, channel_mix
from .pico_backbone import PicoBackbone
from .flash_backbone import FlashBackbone
from .yolov8_backbone import YOLOv8Backbone
from .yolov9_backbone import YOLOv9Backbone
from .yolov10_backbone import YOLOv10Backbone
from .yolov11_backbone import YOLOv11Backbone
from .yolox_backbone import YOLOXBackbone
from flashdet.registry import BACKBONES

BACKBONES.register("LiteBackbone")(LiteBackbone)
BACKBONES.register("PicoBackbone")(PicoBackbone)
BACKBONES.register("FlashBackbone")(FlashBackbone)
BACKBONES.register("YOLOv8Backbone")(YOLOv8Backbone)
BACKBONES.register("YOLOv9Backbone")(YOLOv9Backbone)
BACKBONES.register("YOLOv10Backbone")(YOLOv10Backbone)
BACKBONES.register("YOLOv11Backbone")(YOLOv11Backbone)
BACKBONES.register("YOLOXBackbone")(YOLOXBackbone)

__all__ = [
    "LiteBackbone",
    "PicoBackbone",
    "FlashBackbone",
    "YOLOv8Backbone",
    "YOLOv9Backbone",
    "YOLOv10Backbone",
    "YOLOv11Backbone",
    "YOLOXBackbone",
    "LiteUnit",
    "channel_mix",
]
