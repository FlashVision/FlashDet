"""KalmanPredictor — constant-velocity Kalman filter.

The standard predictor used by SORT, DeepSORT, ByteTrack, etc.
State: [cx, cy, w, h, vx, vy, vw, vh].

Reference:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
    Wojke et al., "Deep SORT", ICIP 2017.
"""

from __future__ import annotations

import numpy as np

from flashdet.trackers.core.predictors.base import BasePredictor
from flashdet.trackers.core.kalman import KalmanFilter
from flashdet.trackers.matching import xyxy_to_cxywh, cxywh_to_xyxy


class KalmanPredictor(BasePredictor):
    """Constant-velocity Kalman filter predictor."""

    def __init__(self):
        self._kf = KalmanFilter()
        self.mean: np.ndarray = np.zeros(8)
        self.covariance: np.ndarray = np.eye(8)

    def initiate(self, bbox: np.ndarray):
        self.mean, self.covariance = self._kf.initiate(xyxy_to_cxywh(bbox))

    def predict(self) -> np.ndarray:
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)
        return cxywh_to_xyxy(self.mean[:4])

    def update(self, bbox: np.ndarray):
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, xyxy_to_cxywh(bbox),
        )
