"""High-level vision solutions built on FlashDet detection + tracking.

All solutions inherit from ``BaseSolution`` and share a common interface:
    - ``process_frame(frame)`` → ``(annotated_frame, results)``
    - ``get_results()``        → latest results dict
    - ``reset()``              → clear internal state

Solutions (ordered by category)
-------------------------------

**Counting & Analytics**
    ObjectCounter        — Count objects crossing a line
    RegionCounter        — Count objects inside polygon regions
    QueueManager         — Monitor queues with wait-time estimation
    CrowdDensity         — Grid-based crowd density estimation
    AnalyticsDashboard   — Aggregate detection statistics over time

**Motion & Tracking**
    SpeedEstimator       — Estimate object speed from track history
    TrajectoryVisualizer — Draw coloured motion trails behind tracks
    TrafficFlow          — Direction-aware traffic flow analysis
    DwellTimeAnalyzer    — Measure time objects spend in zones

**Spatial**
    DistanceCalculator   — Compute real-world distances between objects
    ParkingManager       — Track parking spot occupancy

**Safety & Privacy**
    SecurityAlarm        — Alert when objects enter restricted zones
    ObjectBlurrer        — Blur / anonymize detected objects

**Fitness**
    WorkoutMonitor       — Count exercise repetitions

**Utilities**
    LiveInference        — Real-time webcam/video detection wrapper
    ObjectCropper        — Crop and save detected objects
"""

from flashdet.solutions._base import BaseSolution
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
from flashdet.solutions.trajectory import TrajectoryVisualizer
from flashdet.solutions.object_blurrer import ObjectBlurrer
from flashdet.solutions.crowd_density import CrowdDensity
from flashdet.solutions.dwell_time import DwellTimeAnalyzer
from flashdet.solutions.traffic_flow import TrafficFlow
from flashdet.solutions.object_cropper import ObjectCropper

__all__ = [
    # Base
    "BaseSolution",
    # Counting & Analytics
    "ObjectCounter",
    "RegionCounter",
    "QueueManager",
    "CrowdDensity",
    "AnalyticsDashboard",
    # Motion & Tracking
    "SpeedEstimator",
    "TrajectoryVisualizer",
    "TrafficFlow",
    "DwellTimeAnalyzer",
    # Spatial
    "DistanceCalculator",
    "ParkingManager",
    # Safety & Privacy
    "SecurityAlarm",
    "ObjectBlurrer",
    # Fitness
    "WorkoutMonitor",
    # Heatmap
    "Heatmap",
    # Utilities
    "LiveInference",
    "ObjectCropper",
]
