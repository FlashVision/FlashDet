"""Multi-object tracking for FlashDet."""

from flashdet.trackers.byte_tracker import ByteTracker
from flashdet.trackers.sort_tracker import SORTTracker
from flashdet.trackers.bot_sort import BoTSORT
from flashdet.registry import TRACKERS

TRACKERS.register("ByteTracker")(ByteTracker)
TRACKERS.register("SORTTracker")(SORTTracker)
TRACKERS.register("BoTSORT")(BoTSORT)

__all__ = ["ByteTracker", "SORTTracker", "BoTSORT"]
