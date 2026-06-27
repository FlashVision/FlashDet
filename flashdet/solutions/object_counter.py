"""ObjectCounter — count objects crossing a defined line."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class ObjectCounter(BaseSolution):
    """Count objects crossing a defined line.

    The line splits the frame into two half-planes.  When a track's centre
    crosses from one side to the other, the *in* or *out* counter is
    incremented (direction is determined by the cross-product sign).

    Parameters
    ----------
    predictor : object
        A FlashDet ``Predictor`` (or any callable that accepts an image and
        returns detections as an Nx6 array ``[x1, y1, x2, y2, score, cls]``).
    tracker : FlashTracker | None
        Multi-object tracker.  Defaults to a fresh ``FlashTracker()``.
    line_points : list[tuple[int, int]] | None
        Two endpoints ``[(x1, y1), (x2, y2)]`` defining the counting line.
        If *None*, a horizontal line at the vertical midpoint of the first
        frame is used.
    classes : list[int] | None
        If set, only count objects whose class id is in this list.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        line_points: Optional[List[Tuple[int, int]]] = None,
        classes: Optional[List[int]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.line_points = line_points

        self.in_count: int = 0
        self.out_count: int = 0
        self._track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._side_map: Dict[int, int] = {}

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        if self.line_points is None:
            h = frame.shape[0]
            self.line_points = [(0, h // 2), (frame.shape[1], h // 2)]

        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)

            if not self._filter_class(cls):
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self._track_history[tid].append((cx, cy))

            side = self._point_side(cx, cy)
            if tid in self._side_map:
                prev_side = self._side_map[tid]
                if prev_side != side and prev_side != 0 and side != 0:
                    if side > 0:
                        self.in_count += 1
                    else:
                        self.out_count += 1
            self._side_map[tid] = side

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"ID:{tid}",
                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        pt1, pt2 = self.line_points
        cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"In:{self.in_count}  Out:{self.out_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )

        return annotated, self.get_counts()

    def get_counts(self) -> Dict[str, int]:
        """Return current in/out/total counts."""
        return {
            "in": self.in_count,
            "out": self.out_count,
            "total": self.in_count + self.out_count,
        }

    def get_results(self) -> Dict[str, int]:
        return self.get_counts()

    def reset(self):
        super().reset()
        self.in_count = 0
        self.out_count = 0
        self._track_history.clear()
        self._side_map.clear()

    def _point_side(self, px: float, py: float) -> int:
        (ax, ay), (bx, by) = self.line_points
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        return 0
