"""Tracker algorithm implementations.

Available trackers (ordered by complexity):

- SortTracker      — Pure IoU + Kalman (fastest, simplest)
- ByteTracker      — Two-stage association with low-conf dets
- OCSortTracker    — Observation-centric SORT, handles occlusion
- DeepSortTracker  — Mahalanobis gating + cosine cascade + ReID
- BoTSortTracker   — ReID + camera motion compensation (BoT-SORT)
- StrongSortTracker — Enhanced DeepSORT + EMA + ECC alignment
"""

from flashdet.trackers.algorithms.sort import SortTracker
from flashdet.trackers.algorithms.bytetrack import ByteTracker
from flashdet.trackers.algorithms.ocsort import OCSortTracker
from flashdet.trackers.algorithms.deepsort import DeepSortTracker
from flashdet.trackers.algorithms.botsort import BoTSortTracker
from flashdet.trackers.algorithms.strongsort import StrongSortTracker

__all__ = [
    "SortTracker",
    "ByteTracker",
    "OCSortTracker",
    "DeepSortTracker",
    "BoTSortTracker",
    "StrongSortTracker",
]
