"""MedianFlowPredictor — sparse optical flow based bbox prediction.

Samples a grid of points inside the bounding box, tracks them using
Lucas-Kanade optical flow, and takes the median displacement to
predict the box's new position.  Forward-backward error checking
filters out unreliable flow vectors.

Reference:
    Kalal et al., "Forward-Backward Error: Automatic Detection of
    Tracking Failures", ICPR 2010.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from flashdet.trackers.core.predictors.base import BasePredictor


class MedianFlowPredictor(BasePredictor):
    """Median-flow predictor using sparse Lucas-Kanade optical flow.

    Parameters
    ----------
    grid_size : int
        NxN grid of points sampled inside the bounding box.
    fb_threshold : float
        Maximum forward-backward error to accept a flow vector.
    """

    def __init__(self, grid_size: int = 5, fb_threshold: float = 5.0):
        self._grid_size = grid_size
        self._fb_threshold = fb_threshold
        self._prev_grey: Optional[np.ndarray] = None
        self._curr_grey: Optional[np.ndarray] = None
        self._bbox = np.zeros(4, dtype=np.float64)

        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    @property
    def needs_frame(self) -> bool:
        return True

    def set_frame(self, frame: np.ndarray):
        self._prev_grey = self._curr_grey
        self._curr_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def initiate(self, bbox: np.ndarray):
        self._bbox = bbox.astype(np.float64).copy()

    def predict(self) -> np.ndarray:
        if self._prev_grey is None or self._curr_grey is None:
            return self._bbox.copy()

        points = self._sample_grid(self._bbox)
        if len(points) == 0:
            return self._bbox.copy()

        pts = points.reshape(-1, 1, 2).astype(np.float32)

        # Forward flow
        next_pts, status_f, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_grey, self._curr_grey, pts, None, **self._lk_params,
        )
        if next_pts is None:
            return self._bbox.copy()

        # Backward flow for error checking
        back_pts, status_b, _ = cv2.calcOpticalFlowPyrLK(
            self._curr_grey, self._prev_grey, next_pts, None, **self._lk_params,
        )
        if back_pts is None:
            return self._bbox.copy()

        # Forward-backward error filter
        fb_err = np.linalg.norm(
            pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1,
        )
        mask = (
            (status_f.ravel() == 1)
            & (status_b.ravel() == 1)
            & (fb_err < self._fb_threshold)
        )

        if mask.sum() < 2:
            return self._bbox.copy()

        src = pts.reshape(-1, 2)[mask]
        dst = next_pts.reshape(-1, 2)[mask]
        displacements = dst - src

        # Median displacement
        dx = np.median(displacements[:, 0])
        dy = np.median(displacements[:, 1])

        # Scale change from median pairwise distance ratio
        scale = self._median_scale(src, dst)

        cx = (self._bbox[0] + self._bbox[2]) / 2 + dx
        cy = (self._bbox[1] + self._bbox[3]) / 2 + dy
        w = (self._bbox[2] - self._bbox[0]) * scale
        h = (self._bbox[3] - self._bbox[1]) * scale
        w = max(w, 1.0)
        h = max(h, 1.0)

        self._bbox = np.array([
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
        ], dtype=np.float64)
        return self._bbox.copy()

    def update(self, bbox: np.ndarray):
        self._bbox = bbox.astype(np.float64).copy()

    def _sample_grid(self, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 2), dtype=np.float32)
        xs = np.linspace(x1, x2, self._grid_size + 2)[1:-1]
        ys = np.linspace(y1, y2, self._grid_size + 2)[1:-1]
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        return grid.astype(np.float32)

    @staticmethod
    def _median_scale(src: np.ndarray, dst: np.ndarray) -> float:
        """Estimate scale change from pairwise distances."""
        n = len(src)
        if n < 2:
            return 1.0
        ratios = []
        for i in range(n):
            for j in range(i + 1, n):
                d_src = np.linalg.norm(src[i] - src[j])
                d_dst = np.linalg.norm(dst[i] - dst[j])
                if d_src > 1.0:
                    ratios.append(d_dst / d_src)
        if len(ratios) == 0:
            return 1.0
        return float(np.median(ratios))
