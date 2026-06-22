from .shufflenet import ShuffleNetV2, ShuffleUnit, channel_shuffle
from .resnet import ResNetBackbone, ResNetMultiScaleBackbone
from .text_encoder import TextEncoder
from .yolov9_backbone import YOLOv9Backbone
from .yolov10_backbone import YOLOv10Backbone
from .yolov11_backbone import YOLOv11Backbone
from flashdet.registry import BACKBONES

BACKBONES.register("ShuffleNetV2")(ShuffleNetV2)
BACKBONES.register("ResNet")(ResNetBackbone)
BACKBONES.register("ResNetMultiScale")(ResNetMultiScaleBackbone)
BACKBONES.register("TextEncoder")(TextEncoder)
BACKBONES.register("YOLOv9Backbone")(YOLOv9Backbone)
BACKBONES.register("YOLOv10Backbone")(YOLOv10Backbone)
BACKBONES.register("YOLOv11Backbone")(YOLOv11Backbone)

__all__ = [
    "ShuffleNetV2",
    "ShuffleUnit",
    "channel_shuffle",
    "ResNetBackbone",
    "ResNetMultiScaleBackbone",
    "TextEncoder",
    "YOLOv9Backbone",
    "YOLOv10Backbone",
    "YOLOv11Backbone",
]
