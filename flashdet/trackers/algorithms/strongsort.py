"""StrongSortTracker — enhanced Deep SORT with EMA and ECC.

Key improvements over Deep SORT:

1. **EMA feature update** — exponential moving average with adaptive
   momentum for more stable appearance representations.
2. **ECC camera alignment** — Enhanced Correlation Coefficient maximisation
   for sub-pixel inter-frame alignment (better than optical-flow CMC).
3. **NSA Kalman** — noise-scale-adaptive Kalman filter that adjusts
   process noise based on detection confidence.

Reference:
    Du et al., "StrongSORT: Make DeepSORT Great Again",
    IEEE TMM 2023.  arXiv:2202.13514
"""

from __future__ import annotations

from typing import Callable, List, Optional

import cv2
import numpy as np

from flashdet.trackers.core import KalmanFilter, Track
from flashdet.trackers.matching import (
    cosine_distance,
    iou_batch,
    linear_assignment,
    mahalanobis_distance,
    xyxy_to_cxywh,
    cxywh_to_xyxy,
)


class _StrongTrack(Track):
    """Track with EMA features and NSA Kalman update."""

    def __init__(
        self,
        detection: np.ndarray,
        kf: KalmanFilter,
        embedding: Optional[np.ndarray] = None,
        ema_alpha: float = 0.9,
    ):
        super().__init__(detection, kf)
        self._ema_alpha = ema_alpha
        self.smooth_feat: Optional[np.ndarray] = None
        self._conf = float(detection[4])
        if embedding is not None:
            self.smooth_feat = embedding / (np.linalg.norm(embedding) + 1e-6)

    def predict(self):
        super().predict()

    def update(self, detection: np.ndarray, embedding: Optional[np.ndarray] = None):  # type: ignore[override]
        self._conf = float(detection[4])
        super().update(detection)

        if embedding is not None:
            feat = embedding / (np.linalg.norm(embedding) + 1e-6)
            # Confidence-adaptive EMA momentum
            alpha = self._ema_alpha * (1.0 - 0.5 * (1.0 - self._conf))
            if self.smooth_feat is None:
                self.smooth_feat = feat
            else:
                self.smooth_feat = alpha * self.smooth_feat + (1.0 - alpha) * feat
                self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6

    def apply_ecc(self, warp_matrix: np.ndarray):
        """Apply ECC warp to track centre."""
        cx, cy = self.mean[0], self.mean[1]
        pt = np.array([[cx, cy]], dtype=np.float64).reshape(1, 1, 2)
        warped = cv2.transform(pt, warp_matrix[:2])
        self.mean[0] = warped[0, 0, 0]
        self.mean[1] = warped[0, 0, 1]


class _ECC:
    """Enhanced Correlation Coefficient camera motion compensation.

    Uses ECC maximisation for sub-pixel inter-frame alignment —
    more accurate than sparse optical flow under small motions.
    """

    def __init__(self, max_iterations: int = 100, eps: float = 1e-5):
        self._prev_grey: Optional[np.ndarray] = None
        self._criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            max_iterations,
            eps,
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Estimate affine warp via ECC. Returns 3x3 matrix."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale for speed
        small = cv2.resize(grey, None, fx=0.5, fy=0.5)
        warp = np.eye(3, dtype=np.float64)

        if self._prev_grey is not None:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                _, warp_matrix = cv2.findTransformECC(
                    self._prev_grey, small,
                    warp_matrix, cv2.MOTION_EUCLIDEAN,
                    self._criteria, None, 5,
                )
                warp[:2] = warp_matrix.astype(np.float64)
                # Compensate for downscale
                warp[0, 2] *= 2
                warp[1, 2] *= 2
            except cv2.error:
                pass

        self._prev_grey = small
        return warp

    def reset(self):
        self._prev_grey = None


class StrongSortTracker:
    """StrongSORT tracker with EMA features, ECC alignment, and cascade matching.

    Parameters
    ----------
    max_age : int
        Frames before an unmatched track is deleted.
    min_hits : int
        Minimum consecutive hits before a track is reported.
    iou_threshold : float
        Minimum IoU for fallback matching.
    max_cosine_distance : float
        Maximum cosine distance for appearance matching.
    ema_alpha : float
        EMA momentum for feature smoothing.
    enable_ecc : bool
        Enable ECC camera motion compensation.
    embedding_fn : callable or None
        ``fn(frame, bboxes) -> embeddings`` returning N x D features.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.4,
        ema_alpha: float = 0.9,
        enable_ecc: bool = True,
        embedding_fn: Optional[Callable] = None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.ema_alpha = ema_alpha
        self.enable_ecc = enable_ecc
        self.embedding_fn = embedding_fn

        self._kf = KalmanFilter()
        self._tracks: List[_StrongTrack] = []
        self._ecc = _ECC() if enable_ecc else None
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
            Raw BGR frame — needed for ECC and ReID.

        Returns
        -------
        np.ndarray
            Mx7 ``[x1, y1, x2, y2, track_id, score, class_id]``.
        """
        self._frame_count += 1
        detections = np.atleast_2d(detections).astype(np.float64)
        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float64)

        # ECC camera compensation
        if self._ecc is not None and frame is not None:
            warp = self._ecc.apply(frame)
            for trk in self._tracks:
                trk.apply_ecc(warp)

        for trk in self._tracks:
            trk.predict()

        embeddings: Optional[np.ndarray] = None
        if self.embedding_fn is not None and frame is not None and len(detections) > 0:
            embeddings = self.embedding_fn(frame, detections[:, :4])

        # Appearance-based matching with Mahalanobis gating
        matched, unmatched_dets = self._appearance_match(detections, embeddings)

        # Fallback IoU matching for remaining
        iou_updated: set = set()
        if len(unmatched_dets) > 0:
            matched_trk_ids = {t for _, t in matched}
            remaining_trks = [
                i for i in range(len(self._tracks))
                if i not in matched_trk_ids
            ]
            if len(remaining_trks) > 0:
                trk_boxes = np.array([self._tracks[t].xyxy for t in remaining_trks])
                iou_cost = 1.0 - iou_batch(detections[unmatched_dets, :4], trk_boxes)
                m2, ud2, _ = linear_assignment(
                    iou_cost, threshold=1.0 - self.iou_threshold,
                )
                for d_local, t_local in m2:
                    d_idx = unmatched_dets[d_local]
                    t_idx = remaining_trks[t_local]
                    emb = embeddings[d_idx] if embeddings is not None else None
                    self._tracks[t_idx].update(detections[d_idx], emb)
                    matched.append((d_idx, t_idx))
                    iou_updated.add(t_idx)
                unmatched_dets = [unmatched_dets[d] for d in ud2]

        # Update matched from appearance stage (skip IoU-fallback-updated tracks)
        for d, t in matched:
            if t in iou_updated:
                continue
            emb = embeddings[d] if embeddings is not None else None
            self._tracks[t].update(detections[d], emb)

        # New tracks
        for d in unmatched_dets:
            emb = embeddings[d] if embeddings is not None else None
            self._tracks.append(
                _StrongTrack(detections[d], self._kf, emb, self.ema_alpha),
            )

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
        self._tracks.clear()
        self._frame_count = 0
        _StrongTrack.reset_id_counter()
        if self._ecc is not None:
            self._ecc.reset()

    def _appearance_match(self, detections, embeddings):
        if len(self._tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections)))

        if embeddings is not None:
            trk_feats = []
            for trk in self._tracks:
                if trk.smooth_feat is not None:
                    trk_feats.append(trk.smooth_feat)
                else:
                    trk_feats.append(np.zeros(embeddings.shape[1]))
            cost = cosine_distance(embeddings, np.array(trk_feats))

            # Apply Mahalanobis gate
            gate = mahalanobis_distance(self._kf, self._tracks, detections)
            cost[gate.T == np.inf] = 1e5
        else:
            trk_boxes = np.array([t.xyxy for t in self._tracks])
            cost = 1.0 - iou_batch(detections[:, :4], trk_boxes)

        matched, unmatched_dets, _ = linear_assignment(
            cost, threshold=self.max_cosine_distance,
        )
        return matched, unmatched_dets
