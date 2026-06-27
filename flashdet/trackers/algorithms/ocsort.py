"""OCSortTracker — Observation-Centric SORT for robust occlusion handling.

Key improvements over SORT:

1. **Observation-Centric Re-update (ORU)** — when a track is re-matched
   after being lost, the Kalman state is corrected using the stored
   last observation instead of the drifted prediction.
2. **Observation-Centric Momentum (OCM)** — uses the direction and speed
   between the last two observations as a motion cue in the cost matrix.
3. **Velocity consistency** — penalises matches where the detection
   motion direction contradicts the track's velocity.

Reference:
    Cao et al., "Observation-Centric SORT: Rethinking SORT for Robust
    Multi-Object Tracking", CVPR 2023.  arXiv:2203.14360
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from flashdet.trackers.core import KalmanFilter, Track
from flashdet.trackers.matching import iou_batch, linear_assignment


class _OCTrack(Track):
    """Track with observation history for OC-SORT."""

    def __init__(self, detection: np.ndarray, kf: KalmanFilter):
        super().__init__(detection, kf)
        self.last_observation: np.ndarray = detection[:4].copy()
        self.prev_observation: Optional[np.ndarray] = None
        self.observations: List[tuple] = []
        self._obs_frame: int = 0
        self._frozen_mean: Optional[np.ndarray] = None
        self._frozen_cov: Optional[np.ndarray] = None

    def predict(self):
        if self.time_since_update > 0 and self._frozen_mean is None:
            self._frozen_mean = self.mean.copy()
            self._frozen_cov = self.covariance.copy()
        super().predict()

    def update(self, detection: np.ndarray):
        if self.time_since_update >= 1 and self._frozen_mean is not None:
            self.mean = self._frozen_mean
            self.covariance = self._frozen_cov
            # Replay only observations recorded after the freeze point
            replay_after = self._obs_frame
            replay_obs = [
                (f, obs) for f, obs in self.observations if f > replay_after
            ]
            for _, obs in replay_obs:
                super().update(
                    np.array([obs[0], obs[1], obs[2], obs[3],
                              detection[4], detection[5]])
                )

        self._frozen_mean = None
        self._frozen_cov = None

        self.prev_observation = self.last_observation.copy()
        self.last_observation = detection[:4].copy()
        self._obs_frame += self.time_since_update + 1
        self.observations.append((self._obs_frame, detection[:4].copy()))
        super().update(detection)

    @property
    def velocity(self) -> Optional[np.ndarray]:
        """Velocity vector between last two observations (cx, cy)."""
        if self.prev_observation is None:
            return None
        curr_cx = (self.last_observation[0] + self.last_observation[2]) / 2
        curr_cy = (self.last_observation[1] + self.last_observation[3]) / 2
        prev_cx = (self.prev_observation[0] + self.prev_observation[2]) / 2
        prev_cy = (self.prev_observation[1] + self.prev_observation[3]) / 2
        return np.array([curr_cx - prev_cx, curr_cy - prev_cy], dtype=np.float64)


class OCSortTracker:
    """Observation-Centric SORT tracker.

    Parameters
    ----------
    max_age : int
        Maximum frames a track survives without a detection.
    min_hits : int
        Minimum consecutive hits before a track is reported.
    iou_threshold : float
        Minimum IoU for a valid match.
    delta_t : int
        Observation momentum time window.
    velocity_weight : float
        Weight for velocity consistency in the cost matrix (0 to disable).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        delta_t: int = 3,
        velocity_weight: float = 0.2,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.velocity_weight = velocity_weight

        self._kf = KalmanFilter()
        self._tracks: List[_OCTrack] = []
        self._frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """Process one frame.

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

        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        for d, t in matched:
            self._tracks[t].update(detections[d])

        # Second chance: match unmatched detections to lost tracks using
        # last-observation IoU (observation-centric)
        if len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            lost_trks = [self._tracks[t] for t in unmatched_trks]
            obs_boxes = np.array([t.last_observation for t in lost_trks])
            iou_cost = 1.0 - iou_batch(detections[unmatched_dets, :4], obs_boxes)
            m2, ud2, _ = linear_assignment(iou_cost, threshold=1.0 - self.iou_threshold)
            for d_local, t_local in m2:
                d_idx = unmatched_dets[d_local]
                t_idx = unmatched_trks[t_local]
                self._tracks[t_idx].update(detections[d_idx])
            unmatched_dets = [unmatched_dets[d] for d in ud2]

        for d in unmatched_dets:
            self._tracks.append(_OCTrack(detections[d], self._kf))

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
        _OCTrack.reset_id_counter()

    def _associate(self, detections: np.ndarray):
        if len(self._tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self._tracks)))

        trk_boxes = np.array([t.xyxy for t in self._tracks])
        iou_cost = 1.0 - iou_batch(detections[:, :4], trk_boxes)

        # Velocity consistency penalty
        if self.velocity_weight > 0:
            vel_cost = self._velocity_cost(detections)
            cost = iou_cost + self.velocity_weight * vel_cost
        else:
            cost = iou_cost

        return linear_assignment(cost, threshold=1.0 - self.iou_threshold)

    def _velocity_cost(self, detections: np.ndarray) -> np.ndarray:
        """Penalise matches where motion direction is inconsistent."""
        n_det = len(detections)
        n_trk = len(self._tracks)
        cost = np.zeros((n_det, n_trk), dtype=np.float64)

        det_cx = (detections[:, 0] + detections[:, 2]) / 2
        det_cy = (detections[:, 1] + detections[:, 3]) / 2

        for j, trk in enumerate(self._tracks):
            vel = trk.velocity
            if vel is None:
                continue
            vel_norm = np.linalg.norm(vel)
            if vel_norm < 1.0:
                continue

            trk_cx = (trk.last_observation[0] + trk.last_observation[2]) / 2
            trk_cy = (trk.last_observation[1] + trk.last_observation[3]) / 2
            diff = np.stack([det_cx - trk_cx, det_cy - trk_cy], axis=1)
            diff_norm = np.linalg.norm(diff, axis=1) + 1e-6

            # Cosine angle between velocity and displacement
            cos_angle = (diff @ vel) / (diff_norm * vel_norm)
            cost[:, j] = np.maximum(0.0, -cos_angle)

        return cost
