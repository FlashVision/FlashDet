"""BoTSORT — Bag of Tricks for SORT multi-object tracker.

Extends SORT with three key improvements:

1. **ReID feature matching** — appearance-based cosine similarity prevents
   ID switches when IoU alone is ambiguous.
2. **Camera motion compensation (CMC)** — estimates the inter-frame affine
   transform and warps predicted boxes accordingly, improving matching
   under camera motion.
3. **Hybrid cost matrix** — combines IoU distance with embedding distance
   for more robust association.

Reference: Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
Tracking", arXiv 2206.14651, 2022.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Kalman filter (constant-velocity, centre-size parameterisation)
# ---------------------------------------------------------------------------

class _KalmanFilter:
    """Lightweight Kalman filter for [cx, cy, w, h] with constant velocity."""

    def __init__(self):
        self.ndim = 4
        self._F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.eye(4, 8, dtype=np.float64)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
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

    def predict(self, mean: np.ndarray, cov: np.ndarray):
        std = np.array([
            self._std_pos * mean[2], self._std_pos * mean[3],
            self._std_pos * mean[2], self._std_pos * mean[3],
            self._std_vel * mean[2], self._std_vel * mean[3],
            self._std_vel * mean[2], self._std_vel * mean[3],
        ], dtype=np.float64)
        Q = np.diag(std ** 2)
        mean = self._F @ mean
        cov = self._F @ cov @ self._F.T + Q
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        std = np.array([
            self._std_pos * mean[2], self._std_pos * mean[3],
            self._std_pos * mean[2], self._std_pos * mean[3],
        ], dtype=np.float64)
        R = np.diag(std ** 2)
        projected_mean = self._H @ mean
        S = self._H @ cov @ self._H.T + R
        K = np.linalg.solve(S.T, (cov @ self._H.T).T).T
        y = measurement.astype(np.float64) - projected_mean
        new_mean = mean + K @ y
        new_cov = cov - K @ S @ K.T
        return new_mean, new_cov


# ---------------------------------------------------------------------------
# Single track
# ---------------------------------------------------------------------------

class _BoTTrack:
    """A tracked object with Kalman state and optional ReID embedding."""

    _next_id = 1

    def __init__(
        self, det: np.ndarray, kf: _KalmanFilter,
        embedding: Optional[np.ndarray] = None,
        smooth_alpha: float = 0.9,
    ):
        self.kf = kf
        self.track_id = _BoTTrack._next_id
        _BoTTrack._next_id += 1

        self.score = float(det[4])
        self.class_id = int(det[5])
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        measurement = self._xyxy_to_cxywh(det[:4])
        self.mean, self.covariance = kf.initiate(measurement)

        self._smooth_alpha = smooth_alpha
        self.smooth_feat: Optional[np.ndarray] = None
        if embedding is not None:
            self.smooth_feat = embedding / (np.linalg.norm(embedding) + 1e-6)

    @staticmethod
    def _xyxy_to_cxywh(b: np.ndarray) -> np.ndarray:
        w, h = b[2] - b[0], b[3] - b[1]
        return np.array([b[0] + w / 2, b[1] + h / 2, w, h], dtype=np.float64)

    @staticmethod
    def _cxywh_to_xyxy(c: np.ndarray) -> np.ndarray:
        cx, cy, w, h = c
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, det: np.ndarray, embedding: Optional[np.ndarray] = None):
        measurement = self._xyxy_to_cxywh(det[:4])
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, measurement,
        )
        self.score = float(det[4])
        self.class_id = int(det[5])
        self.hits += 1
        self.time_since_update = 0

        if embedding is not None:
            feat = embedding / (np.linalg.norm(embedding) + 1e-6)
            if self.smooth_feat is None:
                self.smooth_feat = feat
            else:
                self.smooth_feat = (
                    self._smooth_alpha * self.smooth_feat
                    + (1.0 - self._smooth_alpha) * feat
                )
                self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6

    def apply_cmc(self, warp_matrix: np.ndarray):
        """Compensate camera motion by warping the predicted position."""
        cx, cy = self.mean[0], self.mean[1]
        pt = np.array([[cx, cy]], dtype=np.float64).reshape(1, 1, 2)
        warped = cv2.transform(pt, warp_matrix[:2])
        self.mean[0] = warped[0, 0, 0]
        self.mean[1] = warped[0, 0, 1]

    @property
    def xyxy(self) -> np.ndarray:
        return self._cxywh_to_xyxy(self.mean[:4])

    def to_output(self) -> np.ndarray:
        box = self.xyxy
        return np.array([
            box[0], box[1], box[2], box[3],
            self.track_id, self.score, self.class_id,
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def _iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0:1], b[:, 0])
    y1 = np.maximum(a[:, 1:2], b[:, 1])
    x2 = np.minimum(a[:, 2:3], b[:, 2])
    y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Camera Motion Compensation (sparse optical flow based)
# ---------------------------------------------------------------------------

class _CMC:
    """Camera Motion Compensation via sparse optical flow + affine estimation."""

    def __init__(self, max_features: int = 500):
        self._prev_grey: Optional[np.ndarray] = None
        self._max_features = max_features

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Estimate inter-frame affine warp and return the 3×3 warp matrix."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        warp = np.eye(3, dtype=np.float64)

        if self._prev_grey is not None:
            prev_pts = cv2.goodFeaturesToTrack(
                self._prev_grey, maxCorners=self._max_features,
                qualityLevel=0.01, minDistance=10,
            )
            if prev_pts is not None and len(prev_pts) >= 4:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_grey, grey, prev_pts, None,
                )
                if next_pts is not None:
                    mask = status.ravel() == 1
                    if mask.sum() >= 4:
                        src = prev_pts[mask].reshape(-1, 2)
                        dst = next_pts[mask].reshape(-1, 2)
                        M, inliers = cv2.estimateAffinePartial2D(
                            src, dst, method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                        )
                        if M is not None:
                            warp[:2] = M

        self._prev_grey = grey
        return warp

    def reset(self):
        self._prev_grey = None


# ---------------------------------------------------------------------------
# BoTSORT tracker
# ---------------------------------------------------------------------------

class BoTSORT:
    """Bag-of-Tricks SORT multi-object tracker.

    Combines IoU and optional ReID embedding distance for association,
    with camera motion compensation for moving-camera scenarios.

    Parameters
    ----------
    max_age : int
        Frames before an unmatched track is deleted.
    min_hits : int
        Minimum consecutive detections before output.
    iou_threshold : float
        Minimum IoU for a valid detection–track match.
    reid_weight : float
        Weight of the ReID cost in the combined cost matrix (0 to 1).
        Set to 0.0 to disable appearance matching.
    enable_cmc : bool
        Enable camera motion compensation.
    embedding_fn : callable | None
        Function ``fn(frame, bboxes) → embeddings`` that returns an
        N×D feature matrix.  If *None*, pure IoU matching is used.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        reid_weight: float = 0.3,
        enable_cmc: bool = True,
        embedding_fn: Optional[Callable] = None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.reid_weight = reid_weight
        self.enable_cmc = enable_cmc
        self.embedding_fn = embedding_fn

        self._kf = _KalmanFilter()
        self._tracks: List[_BoTTrack] = []
        self._cmc = _CMC() if enable_cmc else None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: np.ndarray,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Process one frame.

        Parameters
        ----------
        detections : np.ndarray
            Nx6 ``[x1, y1, x2, y2, score, class_id]``.
        frame : np.ndarray | None
            The raw BGR frame — needed for CMC and ReID.

        Returns
        -------
        np.ndarray
            Mx7 ``[x1, y1, x2, y2, track_id, score, class_id]``.
        """
        self._frame_count += 1
        detections = np.atleast_2d(detections).astype(np.float64)
        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float64)

        # Camera motion compensation
        if self._cmc is not None and frame is not None:
            warp = self._cmc.apply(frame)
            for trk in self._tracks:
                trk.apply_cmc(warp)

        # Predict
        for trk in self._tracks:
            trk.predict()

        # Compute embeddings
        embeddings: Optional[np.ndarray] = None
        if self.embedding_fn is not None and frame is not None and len(detections) > 0:
            embeddings = self.embedding_fn(frame, detections[:, :4])

        # Associate
        matched, unmatched_d, unmatched_t = self._associate(
            detections, embeddings,
        )

        # Update matched
        for d, t in matched:
            emb = embeddings[d] if embeddings is not None else None
            self._tracks[t].update(detections[d], emb)

        # New tracks
        for d in unmatched_d:
            emb = embeddings[d] if embeddings is not None else None
            self._tracks.append(_BoTTrack(detections[d], self._kf, emb))

        # Prune
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

        # Output
        results = []
        for trk in self._tracks:
            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                results.append(trk.to_output())
        if results:
            return np.array(results, dtype=np.float64)
        return np.empty((0, 7), dtype=np.float64)

    def reset(self):
        """Clear all tracks and state."""
        self._tracks.clear()
        self._frame_count = 0
        _BoTTrack._next_id = 1
        if self._cmc is not None:
            self._cmc.reset()

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------

    def _associate(
        self,
        detections: np.ndarray,
        embeddings: Optional[np.ndarray],
    ):
        if len(self._tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self._tracks)))

        trk_boxes = np.array([t.xyxy for t in self._tracks])
        iou_mat = _iou_batch(detections[:, :4], trk_boxes)
        iou_cost = 1.0 - iou_mat

        # Embedding distance (cosine)
        if embeddings is not None and self.reid_weight > 0:
            emb_cost = self._embedding_cost(embeddings)
            cost = (1.0 - self.reid_weight) * iou_cost + self.reid_weight * emb_cost
        else:
            cost = iou_cost

        from scipy.optimize import linear_sum_assignment

        row_idx, col_idx = linear_sum_assignment(cost)

        matched, unmatched_d, unmatched_t = [], [], []
        det_set = set(range(len(detections)))
        trk_set = set(range(len(self._tracks)))

        for r, c in zip(row_idx, col_idx):
            if iou_mat[r, c] < self.iou_threshold:
                unmatched_d.append(r)
                unmatched_t.append(c)
            else:
                matched.append((r, c))
            det_set.discard(r)
            trk_set.discard(c)

        unmatched_d.extend(det_set)
        unmatched_t.extend(trk_set)
        return matched, unmatched_d, unmatched_t

    def _embedding_cost(self, det_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine distance matrix between detection and track embeddings."""
        n_det = len(det_embeddings)
        n_trk = len(self._tracks)
        cost = np.ones((n_det, n_trk), dtype=np.float64)

        det_normed = det_embeddings / (
            np.linalg.norm(det_embeddings, axis=1, keepdims=True) + 1e-6
        )

        for j, trk in enumerate(self._tracks):
            if trk.smooth_feat is not None:
                similarities = det_normed @ trk.smooth_feat
                cost[:, j] = 1.0 - similarities

        return cost
