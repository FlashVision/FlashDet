from .shufflenet import ShuffleNetV2, ShuffleUnit, channel_shuffle
from flashdet.registry import BACKBONES

BACKBONES.register("ShuffleNetV2")(ShuffleNetV2)

__all__ = ["ShuffleNetV2", "ShuffleUnit", "channel_shuffle"]
