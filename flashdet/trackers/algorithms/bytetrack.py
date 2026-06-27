"""ByteTracker — two-stage association using low-confidence detections.

ByteTrack's key insight is that low-confidence detections (which other
trackers discard) are valuable for maintaining tracks through occlusion.

1. **First stage** — match high-confidence detections to tracks via IoU.
2. **Second stage** — match low-confidence detections to remaining
   unmatched tracks, recovering occluded objects.

Reference:
    Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
    Every Detection Box", ECCV 2022.  arXiv:2110.06864
"""

from __future__ import annotations

from typing import List

import numpy as np

from flashdet.trackers.core import KalmanFilter, Track
from flashdet.trackers.matching import iou_batch, linear_assignment


class ByteTracker:
    """ByteTrack-style multi-object tracker with two-stage association.

    Parameters
    ----------
    max_age : int
        Maximum frames a track survives without an associated detection.
    min_hits : int
        Minimum consecutive hits before a track is reported in output.
    iou_threshold : float
        IoU threshold for first-stage (high-confidence) matching.
    low_iou_threshold : float
        IoU threshold for second-stage (low-confidence) matching.
    high_score_threshold : float
        Score threshold separating high- and low-confidence detections.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        low_iou_threshold: float = 0.2,
        high_score_threshold: float = 0.5,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.high_score_threshold = high_score_threshold

        self._kf = KalmanFilter()
        self._tracks: List[Track] = []
        self._frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame with two-stage association.

        Parameters
        ----------
        detections : np.ndarray
            Nx6 ``[x1, y1, x2, y2, score, class_id]``.
            Include ALL detections (both high and low confidence).

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

        if len(detections) > 0:
            scores = detections[:, 4]
            high_mask = scores >= self.high_score_threshold
            dets_high = detections[high_mask]
            dets_low = detections[~high_mask]
        else:
            dets_high = np.empty((0, 6), dtype=np.float64)
            dets_low = np.empty((0, 6), dtype=np.float64)

        # Stage 1: high-confidence detections
        matched_1, unmatched_dets_high, unmatched_trks = self._match(
            dets_high, self._tracks, self.iou_threshold,
        )
        for d, t in matched_1:
            self._tracks[t].update(dets_high[d])

        # Stage 2: low-confidence detections → remaining tracks
        remaining_tracks = [self._tracks[t] for t in unmatched_trks]
        matched_2, _, _ = self._match(
            dets_low, remaining_tracks, self.low_iou_threshold,
        )
        for d, t in matched_2:
            remaining_tracks[t].update(dets_low[d])

        # New tracks from unmatched high-confidence detections
        for d in unmatched_dets_high:
            self._tracks.append(Track(dets_high[d], self._kf))

        # Prune dead tracks
        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self.max_age
        ]

        # Output confirmed tracks
        results = []
        for trk in self._tracks:
            if trk.time_since_update == 0 and trk.hits >= self.min_hits:
                results.append(trk.to_output())

        if results:
            return np.array(results, dtype=np.float64)
        return np.empty((0, 7), dtype=np.float64)

    def reset(self):
        """Clear all tracks and reset state."""
        self._tracks.clear()
        self._frame_count = 0
        Track.reset_id_counter()

    @staticmethod
    def _match(
        detections: np.ndarray,
        tracks: List[Track],
        iou_threshold: float,
    ):
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))

        trk_boxes = np.array([t.xyxy for t in tracks])
        iou_cost = 1.0 - iou_batch(detections[:, :4], trk_boxes)
        return linear_assignment(iou_cost, threshold=1.0 - iou_threshold)
