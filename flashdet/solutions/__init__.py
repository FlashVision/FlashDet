"""High-level vision solutions built on FlashDet detection + tracking."""

from flashdet.solutions.object_counter import ObjectCounter
from flashdet.solutions.speed_estimator import SpeedEstimator
from flashdet.solutions.heatmap import Heatmap
from flashdet.solutions.region_counter import RegionCounter
from flashdet.solutions.queue_manager import QueueManager
from flashdet.solutions.distance_calculator import DistanceCalculator
from flashdet.solutions.parking_manager import ParkingManager
from flashdet.solutions.security_alarm import SecurityAlarm
from flashdet.solutions.workout_monitor import WorkoutMonitor
from flashdet.solutions.live_inference import LiveInference
from flashdet.solutions.analytics_dashboard import AnalyticsDashboard

__all__ = [
    "ObjectCounter", "SpeedEstimator", "Heatmap", "RegionCounter",
    "QueueManager", "DistanceCalculator", "ParkingManager",
    "SecurityAlarm", "WorkoutMonitor", "LiveInference", "AnalyticsDashboard",
]
