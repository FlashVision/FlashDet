"""DeepSortTracker — deep appearance + Mahalanobis-gated association.

Extends SORT with two key mechanisms:

1. **Mahalanobis gating** — uses Kalman covariance to gate impossible
   matches before computing appearance cost.
2. **Cascade matching** — prioritises recently-seen tracks over lost ones,
   reducing ID switches in crowded scenes.

Reference:
    Wojke et al., "Simple Online and Realtime Tracking with a Deep
    Association Metric", ICIP 2017.  arXiv:1703.07402
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from flashdet.trackers.core import KalmanFilter, Track
from flashdet.trackers.matching import (
    cosine_distance,
    iou_batch,
    linear_assignment,
    mahalanobis_distance,
    xyxy_to_cxywh,
)


class _ReIDTrack(Track):
    """Track with exponentially smoothed ReID embedding for DeepSORT."""

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
                    self._ema_alpha * self.smooth_feat
                    + (1.0 - self._ema_alpha) * feat
                )
                self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6


class DeepSortTracker:
    """Deep SORT tracker with Mahalanobis gating and cascade matching.

    Parameters
    ----------
    max_age : int
        Frames before an unmatched track is deleted.
    min_hits : int
        Minimum consecutive hits before a track is reported.
    iou_threshold : float
        Minimum IoU for fallback matching of unconfirmed tracks.
    max_cosine_distance : float
        Maximum cosine distance for appearance matching.
    cascade_depth : int
        Number of cascade levels (matches most-recent tracks first).
    embedding_fn : callable or None
        ``fn(frame, bboxes) -> embeddings`` returning N x D features.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.4,
        cascade_depth: int = 30,
        embedding_fn: Optional[Callable] = None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_cosine_distance = max_cosine_distance
        self.cascade_depth = cascade_depth
        self.embedding_fn = embedding_fn

        self._kf = KalmanFilter()
        self._tracks: List[_ReIDTrack] = []
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
            Raw BGR frame — needed for ReID embedding extraction.

        Returns
        -------
        np.ndarray
            Mx7 ``[x1, y1, x2, y2, track_id, score, class_id]``.
        """
        self._frame_count += 1
        detections = np.atleast_2d(detections).astype(np.float64)
        if detections.size == 0:
            detections = np.empty((0, 6), dtype=np.float64)

        for trk in self._tracks:
            trk.predict()

        embeddings: Optional[np.ndarray] = None
        if self.embedding_fn is not None and frame is not None and len(detections) > 0:
            embeddings = self.embedding_fn(frame, detections[:, :4])

        # Split tracks: confirmed vs tentative
        confirmed = [i for i, t in enumerate(self._tracks) if t.hits >= self.min_hits]
        tentative = [i for i, t in enumerate(self._tracks) if t.hits < self.min_hits]

        # Stage 1: cascade matching on confirmed tracks (appearance + Mahalanobis)
        matched, unmatched_dets = self._cascade_match(
            detections, embeddings, confirmed,
        )

        # Stage 2: IoU matching on tentative tracks + remaining unmatched
        candidate_trks = tentative + [
            i for i in confirmed if i not in {t for _, t in matched}
            and self._tracks[i].time_since_update == 1
        ]
        iou_updated: set = set()
        if len(unmatched_dets) > 0 and len(candidate_trks) > 0:
            trk_boxes = np.array([self._tracks[t].xyxy for t in candidate_trks])
            iou_cost = 1.0 - iou_batch(detections[unmatched_dets, :4], trk_boxes)
            iou_matched, iou_udet, _ = linear_assignment(
                iou_cost, threshold=1.0 - self.iou_threshold,
            )
            for d_local, t_local in iou_matched:
                d_idx = unmatched_dets[d_local]
                t_idx = candidate_trks[t_local]
                emb = embeddings[d_idx] if embeddings is not None else None
                self._tracks[t_idx].update(detections[d_idx], emb)
                matched.append((d_idx, t_idx))
                iou_updated.add(t_idx)
            unmatched_dets = [unmatched_dets[d] for d in iou_udet]

        # Update matched tracks (skip those already updated in IoU fallback)
        for d, t in matched:
            if t in iou_updated:
                continue
            emb = embeddings[d] if embeddings is not None else None
            self._tracks[t].update(detections[d], emb)

        # Create new tracks
        for d in unmatched_dets:
            emb = embeddings[d] if embeddings is not None else None
            self._tracks.append(_ReIDTrack(detections[d], self._kf, emb))

        # Prune dead tracks
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
        self._tracks.clear()
        self._frame_count = 0
        _ReIDTrack.reset_id_counter()

    def _cascade_match(
        self,
        detections: np.ndarray,
        embeddings: Optional[np.ndarray],
        track_indices: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Cascade matching: match recent tracks first."""
        unmatched_dets = list(range(len(detections)))
        all_matched: List[Tuple[int, int]] = []

        if len(track_indices) == 0 or len(detections) == 0:
            return all_matched, unmatched_dets

        # Mahalanobis gate
        gate = mahalanobis_distance(
            self._kf,
            [self._tracks[i] for i in track_indices],
            detections,
        )

        for age in range(self.cascade_depth):
            if len(unmatched_dets) == 0:
                break

            level_trks = [
                i for i in track_indices
                if self._tracks[i].time_since_update == 1 + age
            ]
            if len(level_trks) == 0:
                continue

            # Appearance cost (cosine)
            if embeddings is not None:
                trk_feats = []
                for t in level_trks:
                    feat = self._tracks[t].smooth_feat
                    if feat is not None:
                        trk_feats.append(feat)
                    else:
                        trk_feats.append(np.zeros(embeddings.shape[1]))
                cost = cosine_distance(
                    embeddings[unmatched_dets], np.array(trk_feats),
                )
            else:
                # Fall back to IoU
                trk_boxes = np.array([self._tracks[t].xyxy for t in level_trks])
                cost = 1.0 - iou_batch(detections[unmatched_dets, :4], trk_boxes)

            # Apply Mahalanobis gate
            for i, t_global in enumerate(level_trks):
                t_gate_idx = track_indices.index(t_global)
                for j, d_idx in enumerate(unmatched_dets):
                    if gate[t_gate_idx, d_idx] == np.inf:
                        cost[j, i] = 1e5

            matched, udet_local, _ = linear_assignment(
                cost, threshold=self.max_cosine_distance,
            )

            for d_local, t_local in matched:
                d_idx = unmatched_dets[d_local]
                t_idx = level_trks[t_local]
                all_matched.append((d_idx, t_idx))

            unmatched_dets = [unmatched_dets[d] for d in udet_local]

        return all_matched, unmatched_dets
