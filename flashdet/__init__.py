"""FlashDet — Ultra-lightweight real-time object detection."""

__version__ = "1.1.0"

from flashdet.models.detector import FlashDet
from flashdet.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashdet.engine.training.trainer import Trainer
from flashdet.engine.evaluation.validator import Validator
from flashdet.engine.inference.predictor import Predictor
from flashdet.engine.export.exporter import Exporter
from flashdet.cfg import get_config
from flashdet.trackers import ByteTracker
from flashdet.solutions import ObjectCounter, SpeedEstimator, Heatmap, RegionCounter
from flashdet.analytics import Benchmark
from flashdet.data.download import download_dataset, list_datasets

__all__ = [
    "FlashDet", "Trainer", "Validator", "Predictor", "Exporter",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "ByteTracker",
    "ObjectCounter", "SpeedEstimator", "Heatmap", "RegionCounter",
    "Benchmark",
    "download_dataset", "list_datasets",
    "__version__",
]
