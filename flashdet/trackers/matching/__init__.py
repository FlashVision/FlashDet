"""Matching utilities for multi-object tracking.

Submodules
----------
iou          — Standard IoU
giou         — GIoU, DIoU, CIoU (Rezatofighi/Zheng et al.)
cosine       — Cosine distance for ReID embeddings
mahalanobis  — Mahalanobis distance with Kalman covariance gating
histogram    — Color histogram appearance matching (no DL)
assignment   — Hungarian algorithm wrapper
geometry     — Box coordinate conversions
"""

from flashdet.trackers.matching.iou import iou_batch
from flashdet.trackers.matching.giou import giou_batch, diou_batch, ciou_batch
from flashdet.trackers.matching.cosine import cosine_distance
from flashdet.trackers.matching.mahalanobis import mahalanobis_distance, CHI2_THRESHOLD_95
from flashdet.trackers.matching.histogram import extract_histograms, histogram_distance
from flashdet.trackers.matching.assignment import linear_assignment
from flashdet.trackers.matching.geometry import xyxy_to_cxywh, cxywh_to_xyxy

__all__ = [
    # IoU variants
    "iou_batch",
    "giou_batch",
    "diou_batch",
    "ciou_batch",
    # Appearance
    "cosine_distance",
    "extract_histograms",
    "histogram_distance",
    # Statistical
    "mahalanobis_distance",
    "CHI2_THRESHOLD_95",
    # Assignment
    "linear_assignment",
    # Geometry
    "xyxy_to_cxywh",
    "cxywh_to_xyxy",
]
