"""FlashDet — Ultra-lightweight real-time object detection."""

__version__ = "1.0.0"

from flashdet.models.detector import FlashDet
from flashdet.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashdet.engine.trainer import Trainer
from flashdet.engine.validator import Validator
from flashdet.engine.predictor import Predictor
from flashdet.engine.exporter import Exporter
from flashdet.cfg import get_config
from flashdet.trackers import ByteTracker
from flashdet.solutions import ObjectCounter, SpeedEstimator, Heatmap, RegionCounter
from flashdet.analytics import Benchmark

__all__ = [
    "FlashDet", "Trainer", "Validator", "Predictor", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "ByteTracker",
    "ObjectCounter", "SpeedEstimator", "Heatmap", "RegionCounter",
    "Benchmark",
    "__version__",
]
