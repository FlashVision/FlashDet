"""ByteTracker — IoU-based multi-object tracker with Kalman filtering.

Assigns persistent IDs to detections across video frames using a
constant-velocity Kalman filter for state prediction and the Hungarian
algorithm for data association.
"""

from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity, bbox centre-size parameterisation)
# ---------------------------------------------------------------------------

class _KalmanFilter:
    """Lightweight constant-velocity Kalman filter operating on [cx, cy, w, h]."""

    def __init__(self):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.ndim = 4
        dt = 1.0

        self._motion_mat = np.eye(8, dtype=np.float64)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = dt

        self._update_mat = np.eye(self.ndim, 8, dtype=np.float64)

        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """Create track from unmatched detection measurement [cx, cy, w, h]."""
        mean_pos = measurement.astype(np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = np.array([
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ], dtype=np.float64)
        covariance = np.diag(std ** 2)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Run Kalman prediction step."""
        std = np.array([
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ], dtype=np.float64)
        motion_cov = np.diag(std ** 2)

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Run Kalman correction step."""
        std = np.array([
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ], dtype=np.float64)
        innovation_cov = np.diag(std ** 2)

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T + innovation_cov

        kalman_gain = np.linalg.solve(
            projected_cov.T,
            (covariance @ self._update_mat.T).T,
        ).T
        innovation = measurement.astype(np.float64) - projected_mean

        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance


# ---------------------------------------------------------------------------
# Single-object track
# ---------------------------------------------------------------------------

class _Track:
    """Internal representation of a single tracked object."""

    _next_id = 1

    def __init__(self, detection: np.ndarray, kf: _KalmanFilter):
        self.kf = kf
        self.track_id = _Track._next_id
        _Track._next_id += 1

        measurement = self._xyxy_to_cxywh(detection[:4])
        self.mean, self.covariance = kf.initiate(measurement)

        self.score: float = float(detection[4])
        self.class_id: int = int(detection[5])
        self.hits: int = 1
        self.age: int = 0
        self.time_since_update: int = 0

    # -- coordinate helpers ------------------------------------------------

    @staticmethod
    def _xyxy_to_cxywh(xyxy: np.ndarray) -> np.ndarray:
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        cx = xyxy[0] + w / 2
        cy = xyxy[1] + h / 2
        return np.array([cx, cy, w, h], dtype=np.float64)

    @staticmethod
    def _cxywh_to_xyxy(cxywh: np.ndarray) -> np.ndarray:
        cx, cy, w, h = cxywh
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)

    # -- lifecycle ---------------------------------------------------------

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: np.ndarray):
        measurement = self._xyxy_to_cxywh(detection[:4])
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        self.score = float(detection[4])
        self.class_id = int(detection[5])
        self.hits += 1
        self.time_since_update = 0

    @property
    def xyxy(self) -> np.ndarray:
        return self._cxywh_to_xyxy(self.mean[:4])

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3  # overridden by ByteTracker.min_hits at output time

    def to_output(self) -> np.ndarray:
        """Return [x1, y1, x2, y2, track_id, score, class_id]."""
        box = self.xyxy
        return np.array([
            box[0], box[1], box[2], box[3],
            self.track_id, self.score, self.class_id,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bboxes (xyxy format).

    Returns shape (len(bb_test), len(bb_gt)).
    """
    x1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0])
    y1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1])
    x2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2])
    y2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])
    union = area_test[:, None] + area_gt[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """IoU-based multi-object tracker with Kalman filtering.

    Parameters
    ----------
    max_age : int
        Maximum frames a track is kept alive without an associated detection.
    min_hits : int
        Minimum consecutive hits before a track is reported in the output.
    iou_threshold : float
        Minimum IoU for a valid detection–track association.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._kf = _KalmanFilter()
        self._tracks: List[_Track] = []
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame of detections and return active tracks.

        Parameters
        ----------
        detections : np.ndarray
            Nx6 array where each row is ``[x1, y1, x2, y2, score, class_id]``.
            Pass an empty (0×6) array when there are no detections.

        Returns
        -------
        np.ndarray
            Mx7 array where each row is
            ``[x1, y1, x2, y2, track_id, score, class_id]``.
        """
        self._frame_count += 1
        detections = np.atleast_2d(detections).astype(np.float64)
        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float64)

        # --- predict existing tracks ---
        for trk in self._tracks:
            trk.predict()

        # --- match detections to tracks (Hungarian on IoU cost) ---
        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        # --- update matched tracks ---
        for det_idx, trk_idx in matched:
            self._tracks[trk_idx].update(detections[det_idx])

        # --- create new tracks for unmatched detections ---
        for det_idx in unmatched_dets:
            self._tracks.append(_Track(detections[det_idx], self._kf))

        # --- remove dead tracks ---
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

        # --- build output (only confirmed tracks) ---
        results = []
        for trk in self._tracks:
            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                results.append(trk.to_output())

        if results:
            return np.array(results, dtype=np.float64)
        return np.empty((0, 7), dtype=np.float64)

    def reset(self):
        """Clear all tracks and reset the frame counter."""
        self._tracks.clear()
        self._frame_count = 0
        _Track._next_id = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _associate(self, detections: np.ndarray):
        """Match detections to existing tracks using IoU + Hungarian."""
        if len(self._tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self._tracks)))

        track_boxes = np.array([t.xyxy for t in self._tracks])
        iou_matrix = _iou_batch(detections[:, :4], track_boxes)

        # Lazy import of scipy to avoid hard dependency at module level
        from scipy.optimize import linear_sum_assignment  # noqa: F811

        cost = 1.0 - iou_matrix
        row_indices, col_indices = linear_sum_assignment(cost)

        matched, unmatched_dets, unmatched_trks = [], [], []

        det_set = set(range(len(detections)))
        trk_set = set(range(len(self._tracks)))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matched.append((r, c))
            det_set.discard(r)
            trk_set.discard(c)

        unmatched_dets.extend(det_set)
        unmatched_trks.extend(trk_set)

        return matched, unmatched_dets, unmatched_trks
