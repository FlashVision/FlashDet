"""LinearPredictor — linear extrapolation from bbox history.

The simplest predictor: fits a line to the last N centre/size
observations and extrapolates one step forward.  No covariance
or uncertainty modelling.  Useful as a lightweight baseline or
fallback when the Kalman filter diverges.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from flashdet.trackers.core.predictors.base import BasePredictor
from flashdet.trackers.matching import xyxy_to_cxywh, cxywh_to_xyxy


class LinearPredictor(BasePredictor):
    """Linear extrapolation predictor.

    Parameters
    ----------
    history_length : int
        Number of past observations used for fitting.
    """

    def __init__(self, history_length: int = 10):
        self._history_length = history_length
        self._history: deque = deque(maxlen=history_length)
        self._last: np.ndarray = np.zeros(4)

    def initiate(self, bbox: np.ndarray):
        cxywh = xyxy_to_cxywh(bbox)
        self._last = cxywh.copy()
        self._history.clear()
        self._history.append(cxywh.copy())

    def predict(self) -> np.ndarray:
        if len(self._history) < 2:
            return cxywh_to_xyxy(self._last)

        pts = np.array(self._history)
        t = np.arange(len(pts), dtype=np.float64)

        predicted = np.zeros(4, dtype=np.float64)
        t_next = float(len(pts))
        for dim in range(4):
            coeffs = np.polyfit(t, pts[:, dim], deg=1)
            predicted[dim] = np.polyval(coeffs, t_next)

        predicted[2] = max(predicted[2], 1.0)  # width > 0
        predicted[3] = max(predicted[3], 1.0)  # height > 0
        self._last = predicted
        return cxywh_to_xyxy(predicted)

    def update(self, bbox: np.ndarray):
        cxywh = xyxy_to_cxywh(bbox)
        self._last = cxywh.copy()
        self._history.append(cxywh.copy())
