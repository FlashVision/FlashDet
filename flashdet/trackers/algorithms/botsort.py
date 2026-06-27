"""BoTSortTracker — appearance-enhanced tracker with ReID and camera motion compensation.

Key improvements over SORT:

1. **ReID feature matching** — cosine similarity on appearance embeddings
   prevents ID switches when IoU alone is ambiguous.
2. **Camera motion compensation (CMC)** — estimates inter-frame affine
   transform via sparse optical flow and warps predicted boxes.
3. **Hybrid cost matrix** — blends IoU distance with embedding distance.

Reference:
    Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian Tracking",
    arXiv:2206.14651, 2022.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import cv2
import numpy as np

from flashdet.trackers.core import KalmanFilter, Track
from flashdet.trackers.matching import (
    cosine_distance,
    cxywh_to_xyxy,
    iou_batch,
    linear_assignment,
    xyxy_to_cxywh,
)


# ---------------------------------------------------------------------------
# ReID-aware track (extends base Track)
# ---------------------------------------------------------------------------

class _ReIDTrack(Track):
    """Track with exponentially smoothed ReID embedding."""

    def __init__(
        self,
        detection: np.ndarray,
        kf: KalmanFilter,
        embedding: Optional[np.ndarray] = None,
        smooth_alpha: float = 0.9,
    ):
        super().__init__(detection, kf)
        self._smooth_alpha = smooth_alpha
        self.smooth_feat: Optional[np.ndarray] = None
        if embedding is not None:
            self.smooth_feat = embedding / (np.linalg.norm(embedding) + 1e-6)

    def update(self, detection: np.ndarray, embedding: Optional[np.ndarray] = None):  # type: ignore[override]
        super().update(detection)
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
        """Compensate camera motion by warping the predicted centre."""
        cx, cy = self.mean[0], self.mean[1]
        pt = np.array([[cx, cy]], dtype=np.float64).reshape(1, 1, 2)
        warped = cv2.transform(pt, warp_matrix[:2])
        self.mean[0] = warped[0, 0, 0]
        self.mean[1] = warped[0, 0, 1]


# ---------------------------------------------------------------------------
# Camera Motion Compensation
# ---------------------------------------------------------------------------

class _CMC:
    """Sparse-optical-flow-based camera motion compensation."""

    def __init__(self, max_features: int = 500):
        self._prev_grey: Optional[np.ndarray] = None
        self._max_features = max_features

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Estimate inter-frame affine warp. Returns a 3x3 matrix."""
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
                        M, _ = cv2.estimateAffinePartial2D(
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
# BoTSortTracker
# ---------------------------------------------------------------------------

class BoTSortTracker:
    """Appearance-enhanced multi-object tracker.

    Combines IoU and optional ReID embedding distance for association,
    with camera motion compensation for moving-camera scenarios.

    Parameters
    ----------
    max_age : int
        Frames before an unmatched track is deleted.
    min_hits : int
        Minimum consecutive detections before a track is reported.
    iou_threshold : float
        Minimum IoU for a valid detection-track match.
    reid_weight : float
        Weight of appearance cost in the combined cost matrix (0 to 1).
        Set to 0.0 to disable appearance matching.
    enable_cmc : bool
        Enable camera motion compensation via optical flow.
    embedding_fn : callable or None
        ``fn(frame, bboxes) -> embeddings`` returning an N x D feature
        matrix.  If None, pure IoU matching is used.
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

        self._kf = KalmanFilter()
        self._tracks: List[_ReIDTrack] = []
        self._cmc = _CMC() if enable_cmc else None
        self._frame_count = 0

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
        frame : np.ndarray or None
            Raw BGR frame — needed for CMC and ReID.

        Returns
        -------
        np.ndarray
            Mx7 ``[x1, y1, x2, y2, track_id, score, class_id]``.
        """
        self._frame_count += 1
        detections = np.atleast_2d(detections).astype(np.float64)
        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float64)

        if self._cmc is not None and frame is not None:
            warp = self._cmc.apply(frame)
            for trk in self._tracks:
                trk.apply_cmc(warp)

        for trk in self._tracks:
            trk.predict()

        embeddings: Optional[np.ndarray] = None
        if self.embedding_fn is not None and frame is not None and len(detections) > 0:
            embeddings = self.embedding_fn(frame, detections[:, :4])

        matched, unmatched_d, _ = self._associate(detections, embeddings)

        for d, t in matched:
            emb = embeddings[d] if embeddings is not None else None
            self._tracks[t].update(detections[d], emb)

        for d in unmatched_d:
            emb = embeddings[d] if embeddings is not None else None
            self._tracks.append(_ReIDTrack(detections[d], self._kf, emb))

        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

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
        _ReIDTrack.reset_id_counter()
        if self._cmc is not None:
            self._cmc.reset()

    def _associate(self, detections: np.ndarray, embeddings: Optional[np.ndarray]):
        if len(self._tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self._tracks)))

        trk_boxes = np.array([t.xyxy for t in self._tracks])
        iou_cost = 1.0 - iou_batch(detections[:, :4], trk_boxes)

        if embeddings is not None and self.reid_weight > 0:
            trk_feats = []
            for trk in self._tracks:
                if trk.smooth_feat is not None:
                    trk_feats.append(trk.smooth_feat)
                else:
                    trk_feats.append(np.zeros(embeddings.shape[1]))
            emb_cost = cosine_distance(embeddings, np.array(trk_feats))
            cost = (1.0 - self.reid_weight) * iou_cost + self.reid_weight * emb_cost
        else:
            cost = iou_cost

        return linear_assignment(cost, threshold=1.0 - self.iou_threshold)
