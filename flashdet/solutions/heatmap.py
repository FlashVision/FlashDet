"""Heatmap — accumulate and visualise detection density over time."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers.byte_tracker import ByteTracker


class Heatmap:
    """Generate a detection heatmap that accumulates over time.

    Each detection's centre contributes a Gaussian blob to the heatmap.
    The result can be blended with the original frame for visualisation.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : ByteTracker | None
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
        tracker: Optional[ByteTracker] = None,
        colormap: int = cv2.COLORMAP_JET,
        radius: int = 40,
        decay: float = 0.995,
        classes: Optional[List[int]] = None,
    ):
        self.predictor = predictor
        self.tracker = tracker
        self.colormap = colormap
        self.radius = radius
        self.decay = decay
        self.classes = classes

        self._heat: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add detections from *frame* to the heatmap.

        Returns
        -------
        overlay : np.ndarray
            Frame blended with the heatmap.
        heatmap : np.ndarray
            The raw BGR heatmap image.
        """
        h, w = frame.shape[:2]
        if self._heat is None:
            self._heat = np.zeros((h, w), dtype=np.float64)

        centres = self._get_centres(frame)

        self._heat *= self.decay

        for cx, cy in centres:
            ix, iy = int(round(cx)), int(round(cy))
            cv2.circle(self._heat, (ix, iy), self.radius, 1.0, -1)

        norm = self._heat.copy()
        max_val = norm.max()
        if max_val > 0:
            norm = (norm / max_val * 255).astype(np.uint8)
        else:
            norm = norm.astype(np.uint8)

        heatmap_bgr = cv2.applyColorMap(norm, self.colormap)
        overlay = cv2.addWeighted(frame, 0.6, heatmap_bgr, 0.4, 0)

        return overlay, heatmap_bgr

    def get_heatmap(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Return the current raw heatmap as a BGR image.

        Parameters
        ----------
        shape : tuple[int, int] | None
            ``(height, width)`` to resize to.  *None* keeps original size.
        """
        if self._heat is None:
            raise RuntimeError("No frames processed yet.")
        norm = self._heat.copy()
        max_val = norm.max()
        if max_val > 0:
            norm = (norm / max_val * 255).astype(np.uint8)
        else:
            norm = norm.astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, self.colormap)
        if shape is not None:
            heatmap = cv2.resize(heatmap, (shape[1], shape[0]))
        return heatmap

    def reset(self):
        """Clear accumulated heatmap."""
        self._heat = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_centres(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        detections = self._run_detector(frame)

        if self.tracker is not None:
            tracks = self.tracker.update(detections)
            data = tracks
        else:
            data = detections

        centres: List[Tuple[float, float]] = []
        for row in data:
            cls = int(row[5]) if self.tracker is None else int(row[6])
            if self.classes is not None and cls not in self.classes:
                continue
            x1, y1, x2, y2 = row[:4]
            centres.append(((x1 + x2) / 2, (y1 + y2) / 2))
        return centres

    def _run_detector(self, frame: np.ndarray) -> np.ndarray:
        result = self.predictor(frame)
        if isinstance(result, np.ndarray):
            return result
        if hasattr(result, "detections"):
            return np.asarray(result.detections, dtype=np.float64)
        return np.empty((0, 6), dtype=np.float64)
