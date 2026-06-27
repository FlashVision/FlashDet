"""EKFPredictor — Extended Kalman Filter for non-linear motion.

Adds acceleration terms to the state vector, allowing the filter to
track objects that speed up, slow down, or change direction.
State: [cx, cy, w, h, vx, vy, vw, vh, ax, ay].

Useful for:
  - Vehicles accelerating/braking
  - Objects with curved trajectories
  - Non-uniform motion patterns
"""

from __future__ import annotations

import numpy as np

from flashdet.trackers.core.predictors.base import BasePredictor
from flashdet.trackers.matching import xyxy_to_cxywh, cxywh_to_xyxy


class EKFPredictor(BasePredictor):
    """Extended Kalman filter with constant-acceleration model.

    Parameters
    ----------
    process_noise : float
        Base process noise scale.
    accel_noise : float
        Acceleration noise scale (higher = more responsive to turns).
    """

    def __init__(self, process_noise: float = 1.0, accel_noise: float = 5.0):
        self._process_noise = process_noise
        self._accel_noise = accel_noise
        # State: [cx, cy, w, h, vx, vy, vw, vh, ax, ay]
        self.mean = np.zeros(10, dtype=np.float64)
        self.covariance = np.eye(10, dtype=np.float64)

        self._H = np.eye(4, 10, dtype=np.float64)  # observe [cx, cy, w, h]

    def initiate(self, bbox: np.ndarray):
        cxywh = xyxy_to_cxywh(bbox)
        self.mean = np.zeros(10, dtype=np.float64)
        self.mean[:4] = cxywh
        self.covariance = np.eye(10, dtype=np.float64) * 10.0
        self.covariance[4:8, 4:8] *= 100.0
        self.covariance[8:, 8:] *= 1000.0

    def predict(self) -> np.ndarray:
        dt = 1.0
        F = np.eye(10, dtype=np.float64)
        # position += velocity * dt + 0.5 * accel * dt^2
        for i in range(4):
            F[i, i + 4] = dt
        F[0, 8] = 0.5 * dt * dt  # cx += 0.5*ax*dt^2
        F[1, 9] = 0.5 * dt * dt  # cy += 0.5*ay*dt^2
        # velocity += accel * dt
        F[4, 8] = dt  # vx += ax*dt
        F[5, 9] = dt  # vy += ay*dt

        Q = np.eye(10, dtype=np.float64) * self._process_noise
        Q[4:8, 4:8] *= 2.0
        Q[8:, 8:] *= self._accel_noise

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q

        self.mean[2] = max(self.mean[2], 1.0)
        self.mean[3] = max(self.mean[3], 1.0)
        return cxywh_to_xyxy(self.mean[:4])

    def update(self, bbox: np.ndarray):
        measurement = xyxy_to_cxywh(bbox).astype(np.float64)
        S = self._H @ self.covariance @ self._H.T + np.eye(4) * 1.0
        K = self.covariance @ self._H.T @ np.linalg.inv(S)
        innovation = measurement - self._H @ self.mean
        self.mean = self.mean + K @ innovation
        self.covariance = (np.eye(10) - K @ self._H) @ self.covariance
