"""Multi-object tracking for FlashDet.

Structure
---------
core/            — Kalman filter, base Track class, bbox predictors
core/predictors/ — Pluggable prediction models (Kalman, EKF, KCF, MedianFlow, Linear)
matching/        — IoU, GIoU/DIoU/CIoU, cosine, Mahalanobis, histogram, assignment
algorithms/      — Tracker implementations

Prediction Models (in core/predictors/)
---------------------------------------
KalmanPredictor      — Standard constant-velocity Kalman filter
LinearPredictor      — Simple linear extrapolation from history
EKFPredictor         — Extended Kalman filter (acceleration model)
MedianFlowPredictor  — Lucas-Kanade sparse optical flow + FB error
KCFPredictor         — Kernelized Correlation Filter (HOG + ridge regression)

Trackers (ordered by complexity)
---------------------------------
SortTracker       — IoU + Kalman (Bewley et al., ICIP 2016)
ByteTracker       — Two-stage low-conf association (Zhang et al., ECCV 2022)
OCSortTracker     — Observation-centric SORT (Cao et al., CVPR 2023)
DeepSortTracker   — Mahalanobis + cosine cascade (Wojke et al., ICIP 2017)
BoTSortTracker    — ReID + camera motion compensation (Aharon et al., 2022)
StrongSortTracker — Enhanced DeepSORT + EMA + ECC (Du et al., TMM 2023)

Backward-compatible aliases:
    FlashTracker      → SortTracker
    MotionTracker     → SortTracker
    AppearanceTracker → BoTSortTracker
"""

from flashdet.trackers.algorithms import (
    SortTracker,
    ByteTracker,
    OCSortTracker,
    DeepSortTracker,
    BoTSortTracker,
    StrongSortTracker,
)
from flashdet.trackers.core import (
    KalmanFilter,
    Track,
    BasePredictor,
    KalmanPredictor,
    LinearPredictor,
    EKFPredictor,
    MedianFlowPredictor,
    KCFPredictor,
)
from flashdet.trackers.matching import (
    iou_batch,
    giou_batch,
    diou_batch,
    ciou_batch,
    cosine_distance,
    mahalanobis_distance,
    extract_histograms,
    histogram_distance,
    linear_assignment,
    xyxy_to_cxywh,
    cxywh_to_xyxy,
)
from flashdet.registry import TRACKERS

# Register all trackers
TRACKERS.register("SortTracker")(SortTracker)
TRACKERS.register("ByteTracker")(ByteTracker)
TRACKERS.register("OCSortTracker")(OCSortTracker)
TRACKERS.register("DeepSortTracker")(DeepSortTracker)
TRACKERS.register("BoTSortTracker")(BoTSortTracker)
TRACKERS.register("StrongSortTracker")(StrongSortTracker)

# Backward-compatible aliases
FlashTracker = SortTracker
MotionTracker = SortTracker
AppearanceTracker = BoTSortTracker

TRACKERS.register("FlashTracker")(FlashTracker)
TRACKERS.register("MotionTracker")(MotionTracker)
TRACKERS.register("AppearanceTracker")(AppearanceTracker)

__all__ = [
    # Algorithms
    "SortTracker",
    "ByteTracker",
    "OCSortTracker",
    "DeepSortTracker",
    "BoTSortTracker",
    "StrongSortTracker",
    # Core
    "KalmanFilter",
    "Track",
    # Predictors
    "BasePredictor",
    "KalmanPredictor",
    "LinearPredictor",
    "EKFPredictor",
    "MedianFlowPredictor",
    "KCFPredictor",
    # Matching
    "iou_batch",
    "giou_batch",
    "diou_batch",
    "ciou_batch",
    "cosine_distance",
    "mahalanobis_distance",
    "extract_histograms",
    "histogram_distance",
    "linear_assignment",
    "xyxy_to_cxywh",
    "cxywh_to_xyxy",
    # Aliases
    "FlashTracker",
    "MotionTracker",
    "AppearanceTracker",
]
