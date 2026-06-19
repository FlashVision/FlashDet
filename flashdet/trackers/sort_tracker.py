"""SORTTracker — Simple Online and Realtime Tracker.

A simpler and faster alternative to ByteTracker.  Uses a constant-velocity
Kalman filter for prediction and the Hungarian algorithm (IoU cost) for
data association, without the two-stage matching that ByteTrack employs.

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity, xyah parameterisation)
# ---------------------------------------------------------------------------

class _KalmanBoxTracker:
    """Track a single object with a constant-velocity Kalman filter.

    State vector: ``[cx, cy, area, aspect_ratio, vcx, vcy, va, var]``.
    The area/aspect-ratio parameterisation improves scale estimation
    compared to raw width/height.
    """

    _count = 0

    def __init__(self, bbox: np.ndarray, score: float, cls: int):
        _KalmanBoxTracker._count += 1
        self.id = _KalmanBoxTracker._count
        self.score = score
        self.class_id = cls
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[np.ndarray] = []

        z = self._xyxy_to_z(bbox)
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4] = z.reshape(4, 1)

        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 1000.0

        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.eye(4, 8, dtype=np.float64)

        self.Q = np.eye(8, dtype=np.float64)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0

    # -- coordinate transforms -------------------------------------------

    @staticmethod
    def _xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1,y1,x2,y2] → [cx,cy,area,aspect_ratio]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        area = w * h
        ar = w / (h + 1e-6)
        return np.array([cx, cy, area, ar], dtype=np.float64)

    @staticmethod
    def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
        """Convert [cx,cy,area,aspect_ratio] → [x1,y1,x2,y2]."""
        cx, cy, area, ar = z.flatten()[:4]
        area = max(area, 1.0)
        w = np.sqrt(area * ar)
        h = area / (w + 1e-6)
        return np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ], dtype=np.float64)

    # -- KF operations ----------------------------------------------------

    def predict(self) -> np.ndarray:
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: np.ndarray, score: float, cls: int):
        self.time_since_update = 0
        self.hits += 1
        self.score = score
        self.class_id = cls

        z = self._xyxy_to_z(bbox).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        eye = np.eye(8, dtype=np.float64)
        self.P = (eye - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        return self._z_to_xyxy(self.x[:4])

    def to_output(self) -> np.ndarray:
        box = self.get_state()
        return np.array([
            box[0], box[1], box[2], box[3],
            self.id, self.score, self.class_id,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _iou_batch(bb_a: np.ndarray, bb_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
    x1 = np.maximum(bb_a[:, 0:1], bb_b[:, 0])
    y1 = np.maximum(bb_a[:, 1:2], bb_b[:, 1])
    x2 = np.minimum(bb_a[:, 2:3], bb_b[:, 2])
    y2 = np.minimum(bb_a[:, 3:4], bb_b[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (bb_a[:, 2] - bb_a[:, 0]) * (bb_a[:, 3] - bb_a[:, 1])
    area_b = (bb_b[:, 2] - bb_b[:, 0]) * (bb_b[:, 3] - bb_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# SORTTracker
# ---------------------------------------------------------------------------

class SORTTracker:
    """Simple Online and Realtime Tracker.

    Uses Kalman-filter prediction + Hungarian-algorithm IoU matching.
    Simpler and faster than ByteTracker (single-stage matching only),
    but less robust in crowded or occluded scenes.

    Parameters
    ----------
    max_age : int
        Frames before an unmatched track is deleted.
    min_hits : int
        Minimum consecutive detections before a track is reported.
    iou_threshold : float
        Minimum IoU for a valid detection–track match.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: List[_KalmanBoxTracker] = []
        self._frame_count: int = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame of detections and return active tracks.

        Parameters
        ----------
        detections : np.ndarray
            Nx6 array ``[x1, y1, x2, y2, score, class_id]``.

        Returns
        -------
        np.ndarray
            Mx7 ``[x1, y1, x2, y2, track_id, score, class_id]``.
        """
        self._frame_count += 1
        detections = np.atleast_2d(detections).astype(np.float64)
        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float64)

        # Predict existing trackers
        predicted_boxes = np.zeros((len(self._trackers), 4), dtype=np.float64)
        for i, trk in enumerate(self._trackers):
            predicted_boxes[i] = trk.predict()

        # Associate
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted_boxes,
        )

        # Update matched trackers
        for d, t in matched:
            self._trackers[t].update(
                detections[d, :4], float(detections[d, 4]), int(detections[d, 5]),
            )

        # Create new trackers for unmatched detections
        for d in unmatched_dets:
            self._trackers.append(
                _KalmanBoxTracker(
                    detections[d, :4],
                    float(detections[d, 4]),
                    int(detections[d, 5]),
                )
            )

        # Prune dead trackers
        self._trackers = [
            t for t in self._trackers if t.time_since_update <= self.max_age
        ]

        # Output confirmed tracks
        results = []
        for trk in self._trackers:
            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                results.append(trk.to_output())

        if results:
            return np.array(results, dtype=np.float64)
        return np.empty((0, 7), dtype=np.float64)

    def reset(self):
        """Clear all tracks."""
        self._trackers.clear()
        self._frame_count = 0
        _KalmanBoxTracker._count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _associate(self, detections: np.ndarray, trk_boxes: np.ndarray):
        if len(self._trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self._trackers)))

        iou_matrix = _iou_batch(detections[:, :4], trk_boxes)

        from scipy.optimize import linear_sum_assignment

        cost = 1.0 - iou_matrix
        row_idx, col_idx = linear_sum_assignment(cost)

        matched, unmatched_d, unmatched_t = [], [], []
        det_set = set(range(len(detections)))
        trk_set = set(range(len(self._trackers)))

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] < self.iou_threshold:
                unmatched_d.append(r)
                unmatched_t.append(c)
            else:
                matched.append((r, c))
            det_set.discard(r)
            trk_set.discard(c)

        unmatched_d.extend(det_set)
        unmatched_t.extend(trk_set)
        return matched, unmatched_d, unmatched_t
