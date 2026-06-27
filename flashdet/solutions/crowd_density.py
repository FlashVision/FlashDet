"""CrowdDensity — grid-based crowd density estimation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class CrowdDensity(BaseSolution):
    """Estimate crowd density by dividing the frame into a grid.

    Each cell tracks how many detected objects (or tracked IDs) have
    their centre inside it.  Cells are colour-coded from green (empty)
    to red (crowded) based on a configurable threshold.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker (uses detections directly if None).
    grid_rows : int
        Number of rows in the density grid.
    grid_cols : int
        Number of columns in the density grid.
    high_density_threshold : int
        Count above which a cell is considered "high density".
    classes : list[int] | None
        Only count these class IDs.
    alpha : float
        Transparency of the density overlay (0-1).
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        grid_rows: int = 6,
        grid_cols: int = 8,
        high_density_threshold: int = 5,
        classes: Optional[List[int]] = None,
        alpha: float = 0.35,
    ):
        super().__init__(predictor, tracker, classes)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.high_density_threshold = high_density_threshold
        self.alpha = alpha
        self._last_grid: Optional[np.ndarray] = None
        self._frame_idx: int = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_idx += 1
        detections = self._detect(frame)

        if self.tracker is not None:
            data = self.tracker.update(detections)
            cls_col = 6
        else:
            data = detections
            cls_col = 5

        h, w = frame.shape[:2]
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols

        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        for row in data:
            cls = int(row[cls_col])
            if not self._filter_class(cls):
                continue
            cx = (row[0] + row[2]) / 2
            cy = (row[1] + row[3]) / 2
            ci = min(int(cy / cell_h), self.grid_rows - 1)
            cj = min(int(cx / cell_w), self.grid_cols - 1)
            grid[ci, cj] += 1

        self._last_grid = grid
        annotated = frame.copy()
        overlay = annotated.copy()

        max_count = max(grid.max(), 1)
        for ri in range(self.grid_rows):
            for ci in range(self.grid_cols):
                count = grid[ri, ci]
                if count == 0:
                    continue
                ratio = min(count / self.high_density_threshold, 1.0)
                # green -> yellow -> red
                r = int(255 * ratio)
                g = int(255 * (1 - ratio))
                color = (0, g, r)

                x1 = int(ci * cell_w)
                y1 = int(ri * cell_h)
                x2 = int((ci + 1) * cell_w)
                y2 = int((ri + 1) * cell_h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.putText(
                    overlay, str(count),
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )

        cv2.addWeighted(overlay, self.alpha, annotated, 1 - self.alpha, 0, annotated)

        # Grid lines
        for ri in range(1, self.grid_rows):
            y = int(ri * cell_h)
            cv2.line(annotated, (0, y), (w, y), (100, 100, 100), 1)
        for ci in range(1, self.grid_cols):
            x = int(ci * cell_w)
            cv2.line(annotated, (x, 0), (x, h), (100, 100, 100), 1)

        total = int(grid.sum())
        cv2.putText(
            annotated, f"Total: {total}  Peak cell: {int(max_count)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        if self._last_grid is None:
            return {"total": 0, "grid": [], "peak_cell": 0}
        return {
            "frame_idx": self._frame_idx,
            "total": int(self._last_grid.sum()),
            "grid": self._last_grid.tolist(),
            "peak_cell": int(self._last_grid.max()),
            "high_density_cells": int(
                (self._last_grid >= self.high_density_threshold).sum()
            ),
        }

    def reset(self):
        super().reset()
        self._last_grid = None
        self._frame_idx = 0
