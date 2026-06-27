"""Base Track class used by all tracker algorithms."""

from __future__ import annotations


import numpy as np

from flashdet.trackers.core.kalman import KalmanFilter
from flashdet.trackers.matching import xyxy_to_cxywh, cxywh_to_xyxy


class Track:
    """Single-object track with Kalman state.

    Subclassed by algorithm-specific tracks that need extra fields
    (e.g. ReID embeddings in BoT-SORT).
    """

    _next_id = 1

    def __init__(self, detection: np.ndarray, kf: KalmanFilter):
        self.kf = kf
        self.track_id = Track._next_id
        Track._next_id += 1

        self.mean, self.covariance = kf.initiate(xyxy_to_cxywh(detection[:4]))
        self.score = float(detection[4])
        self.class_id = int(detection[5])
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        """Run Kalman prediction."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: np.ndarray):
        """Run Kalman correction with a matched detection."""
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, xyxy_to_cxywh(detection[:4]),
        )
        self.score = float(detection[4])
        self.class_id = int(detection[5])
        self.hits += 1
        self.time_since_update = 0

    @property
    def xyxy(self) -> np.ndarray:
        """Current bounding box in [x1, y1, x2, y2] format."""
        return cxywh_to_xyxy(self.mean[:4])

    def to_output(self) -> np.ndarray:
        """Return [x1, y1, x2, y2, track_id, score, class_id]."""
        box = self.xyxy
        return np.array([
            box[0], box[1], box[2], box[3],
            self.track_id, self.score, self.class_id,
        ], dtype=np.float64)

    @classmethod
    def reset_id_counter(cls):
        """Reset the global track ID counter."""
        cls._next_id = 1
