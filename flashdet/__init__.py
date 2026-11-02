"""FlashDet — Ultra-lightweight real-time object detection."""

try:
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version("flashdet")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "1.2.1"

from flashdet.models.detector import FlashDet, build_model
from flashdet.models.lora import apply_lora, apply_qlora, merge_lora_weights
from flashdet.engine.training.trainer import Trainer
from flashdet.engine.evaluation.validator import Validator
from flashdet.engine.inference import Predictor
from flashdet.cfg import get_config
from flashdet.trackers import FlashTracker
from flashdet.solutions import (
    ObjectCounter, SpeedEstimator, Heatmap, RegionCounter,
    QueueManager, DistanceCalculator, ParkingManager,
    SecurityAlarm, WorkoutMonitor, LiveInference, AnalyticsDashboard,
)
from flashdet.analytics import Benchmark
from flashdet.data.download import download_dataset, list_datasets

__all__ = [
    "FlashDet", "build_model", "Trainer", "Validator", "Predictor",
    "apply_lora", "apply_qlora", "merge_lora_weights", "get_config",
    "FlashTracker",
    "ObjectCounter", "SpeedEstimator", "Heatmap", "RegionCounter",
    "QueueManager", "DistanceCalculator", "ParkingManager",
    "SecurityAlarm", "WorkoutMonitor", "LiveInference", "AnalyticsDashboard",
    "Benchmark",
    "download_dataset", "list_datasets",
    "__version__",
]
