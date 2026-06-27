"""TrajectoryVisualizer — draw track trails (motion paths) on video frames."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class TrajectoryVisualizer(BaseSolution):
    """Draw coloured motion trails behind tracked objects.

    Each track accumulates a tail of recent centre positions.  The tail
    is drawn as a polyline that fades from full opacity to transparent,
    giving a smooth motion trail effect.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    tail_length : int
        Maximum number of past positions to keep per track.
    line_thickness : int
        Thickness of the trail polyline.
    fade : bool
        If True, the trail fades towards the older end.
    classes : list[int] | None
        Only draw trails for these class IDs.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        tail_length: int = 60,
        line_thickness: int = 2,
        fade: bool = True,
        classes: Optional[List[int]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.tail_length = tail_length
        self.line_thickness = line_thickness
        self.fade = fade

        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=tail_length)
        )

        rng = np.random.RandomState(42)
        self._palette = [
            tuple(int(c) for c in rng.randint(60, 255, 3))
            for _ in range(200)
        ]

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()
        active_ids: List[int] = []

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)
            if not self._filter_class(cls):
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self._trails[tid].append((int(cx), int(cy)))
            active_ids.append(tid)

            color = self._palette[tid % len(self._palette)]
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                annotated, f"ID:{tid}",
                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

            trail = list(self._trails[tid])
            self._draw_trail(annotated, trail, color)

        # Prune trails for tracks no longer active
        dead_ids = [tid for tid in self._trails if tid not in active_ids]
        for tid in dead_ids:
            del self._trails[tid]

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        return {
            "active_tracks": len(self._trails),
            "trail_lengths": {tid: len(trail) for tid, trail in self._trails.items()},
        }

    def reset(self):
        super().reset()
        self._trails.clear()

    def _draw_trail(
        self,
        img: np.ndarray,
        trail: List[Tuple[int, int]],
        color: Tuple[int, int, int],
    ):
        if len(trail) < 2:
            return

        if not self.fade:
            pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], False, color, self.line_thickness)
            return

        n = len(trail)
        for i in range(1, n):
            alpha = i / n
            c = tuple(int(v * alpha) for v in color)
            thick = max(1, int(self.line_thickness * alpha))
            cv2.line(img, trail[i - 1], trail[i], c, thick)
