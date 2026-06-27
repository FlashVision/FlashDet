"""SortTracker — IoU-based multi-object tracker with Kalman filtering.

Reference:
    Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
    arXiv:1602.00763
"""

from __future__ import annotations

from typing import List

import numpy as np

from flashdet.trackers.core import KalmanFilter, Track
from flashdet.trackers.matching import iou_batch, linear_assignment


class SortTracker:
    """SORT-based multi-object tracker.

    Uses a constant-velocity Kalman filter for motion prediction and the
    Hungarian algorithm with IoU cost for detection-to-track association.

    Parameters
    ----------
    max_age : int
        Maximum frames a track survives without an associated detection.
    min_hits : int
        Minimum consecutive hits before a track is reported in output.
    iou_threshold : float
        Minimum IoU for a valid detection-track match.
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

        self._kf = KalmanFilter()
        self._tracks: List[Track] = []
        self._frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame and return active tracks.

        Parameters
        ----------
        detections : np.ndarray
            Nx6 ``[x1, y1, x2, y2, score, class_id]``.

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

        matched, unmatched_dets, _ = self._associate(detections)

        for d, t in matched:
            self._tracks[t].update(detections[d])

        for d in unmatched_dets:
            self._tracks.append(Track(detections[d], self._kf))

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
        """Clear all tracks and reset state."""
        self._tracks.clear()
        self._frame_count = 0
        Track.reset_id_counter()

    def _associate(self, detections: np.ndarray):
        if len(self._tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self._tracks)))

        track_boxes = np.array([t.xyxy for t in self._tracks])
        iou_cost = 1.0 - iou_batch(detections[:, :4], track_boxes)
        return linear_assignment(iou_cost, threshold=1.0 - self.iou_threshold)
