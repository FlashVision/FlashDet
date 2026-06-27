"""Heatmap — accumulate and visualise detection density over time."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class Heatmap(BaseSolution):
    """Generate a detection heatmap that accumulates over time.

    Each detection's centre contributes a solid circle to the heatmap.
    The result can be blended with the original frame for visualisation.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Optional tracker — when provided the heatmap is built from track
        centres (more stable); otherwise raw detections are used.
    colormap : int
        OpenCV colormap constant (default ``cv2.COLORMAP_JET``).
    radius : int
        Radius (in pixels) of the Gaussian blob added per detection.
    decay : float
        Per-frame multiplicative decay applied to the accumulator so old
        detections fade out.  Set to 1.0 to disable decay.
    classes : list[int] | None
        Only include these class IDs.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        colormap: int = cv2.COLORMAP_JET,
        radius: int = 40,
        decay: float = 0.995,
        classes: Optional[List[int]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self.colormap = colormap
        self.radius = radius
        self.decay = decay
        self._heat: Optional[np.ndarray] = None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        if self._heat is None:
            self._heat = np.zeros((h, w), dtype=np.float64)

        centres = self._get_centres(frame)
        self._heat *= self.decay

        for cx, cy in centres:
            ix, iy = int(round(cx)), int(round(cy))
            cv2.circle(self._heat, (ix, iy), self.radius, 1.0, -1)

        heatmap_bgr = self._normalize_heat()
        overlay = cv2.addWeighted(frame, 0.6, heatmap_bgr, 0.4, 0)
        return overlay, heatmap_bgr

    def get_heatmap(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Return the current raw heatmap as a BGR image."""
        if self._heat is None:
            raise RuntimeError("No frames processed yet.")
        heatmap = self._normalize_heat()
        if shape is not None:
            heatmap = cv2.resize(heatmap, (shape[1], shape[0]))
        return heatmap

    def get_results(self) -> Dict[str, Any]:
        return {"has_heatmap": self._heat is not None}

    def reset(self):
        super().reset()
        self._heat = None

    def _get_centres(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        detections = self._detect(frame)

        if self.tracker is not None:
            tracks = self.tracker.update(detections)
            data = tracks
        else:
            data = detections

        centres: List[Tuple[float, float]] = []
        for row in data:
            cls = int(row[5]) if self.tracker is None else int(row[6])
            if not self._filter_class(cls):
                continue
            x1, y1, x2, y2 = row[:4]
            centres.append(((x1 + x2) / 2, (y1 + y2) / 2))
        return centres

    def _normalize_heat(self) -> np.ndarray:
        norm = self._heat.copy()
        max_val = norm.max()
        if max_val > 0:
            norm = (norm / max_val * 255).astype(np.uint8)
        else:
            norm = norm.astype(np.uint8)
        return cv2.applyColorMap(norm, self.colormap)
