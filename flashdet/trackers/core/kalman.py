"""Constant-velocity Kalman filter for bounding-box tracking.

State vector: [cx, cy, w, h, vx, vy, vw, vh]
Measurement:  [cx, cy, w, h]

Process and measurement noise are scaled relative to the box size,
following the approach in Deep SORT (Wojke et al., 2017).
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Constant-velocity Kalman filter operating on [cx, cy, w, h]."""

    def __init__(self):
        self.ndim = 4
        self._F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.eye(4, 8, dtype=np.float64)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """Create a new track from an initial measurement [cx, cy, w, h]."""
        mean = np.concatenate([measurement, np.zeros(4)]).astype(np.float64)
        std = np.array([
            2 * self._std_pos * measurement[2],
            2 * self._std_pos * measurement[3],
            2 * self._std_pos * measurement[2],
            2 * self._std_pos * measurement[3],
            10 * self._std_vel * measurement[2],
            10 * self._std_vel * measurement[3],
            10 * self._std_vel * measurement[2],
            10 * self._std_vel * measurement[3],
        ], dtype=np.float64)
        return mean, np.diag(std ** 2)

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run the prediction step."""
        std = np.array([
            self._std_pos * mean[2], self._std_pos * mean[3],
            self._std_pos * mean[2], self._std_pos * mean[3],
            self._std_vel * mean[2], self._std_vel * mean[3],
            self._std_vel * mean[2], self._std_vel * mean[3],
        ], dtype=np.float64)
        Q = np.diag(std ** 2)
        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run the correction step."""
        std = np.array([
            self._std_pos * mean[2], self._std_pos * mean[3],
            self._std_pos * mean[2], self._std_pos * mean[3],
        ], dtype=np.float64)
        R = np.diag(std ** 2)
        projected_mean = self._H @ mean
        S = self._H @ covariance @ self._H.T + R
        K = np.linalg.solve(S.T, (covariance @ self._H.T).T).T
        innovation = measurement.astype(np.float64) - projected_mean
        new_mean = mean + K @ innovation
        new_covariance = covariance - K @ S @ K.T
        return new_mean, new_covariance
