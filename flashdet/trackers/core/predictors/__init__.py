"""Bounding-box prediction models for multi-object tracking.

Each predictor follows a common interface:
    - ``initiate(bbox)``  → create internal state from first detection
    - ``predict()``       → predict next bbox, return [x1, y1, x2, y2]
    - ``update(bbox)``    → correct state with matched detection

Available predictors (ordered by complexity):
    LinearPredictor      — Linear extrapolation from bbox history
    KalmanPredictor      — Constant-velocity Kalman filter (standard)
    EKFPredictor         — Extended Kalman filter (non-linear motion)
    MedianFlowPredictor  — Lucas-Kanade sparse optical flow
    KCFPredictor         — Kernelized Correlation Filter
"""

from flashdet.trackers.core.predictors.base import BasePredictor
from flashdet.trackers.core.predictors.kalman_predictor import KalmanPredictor
from flashdet.trackers.core.predictors.linear_predictor import LinearPredictor
from flashdet.trackers.core.predictors.ekf_predictor import EKFPredictor
from flashdet.trackers.core.predictors.median_flow import MedianFlowPredictor
from flashdet.trackers.core.predictors.kcf_predictor import KCFPredictor

__all__ = [
    "BasePredictor",
    "KalmanPredictor",
    "LinearPredictor",
    "EKFPredictor",
    "MedianFlowPredictor",
    "KCFPredictor",
]
