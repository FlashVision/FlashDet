"""SpeedEstimator — estimate object speed from track history."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class SpeedEstimator(BaseSolution):
    """Estimate object speed (px/frame or real-world units) from track history.

    Speed is computed as the Euclidean displacement of a track's centre over a
    sliding window of recent frames.  An optional ``pixels_per_meter`` factor
    and ``fps`` value can convert the result to real-world km/h.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    pixels_per_meter : float
        Calibration factor.  Set to 1.0 to get speed in px/frame.
    fps : float
        Video frame rate — used together with *pixels_per_meter* for km/h.
    window : int
        Number of past positions to keep for the speed calculation.
    classes : list[int] | None
        Only estimate speed for these class IDs.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        pixels_per_meter: float = 1.0,
        fps: float = 30.0,
        window: int = 10,
        classes: Optional[List[int]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.window = window

        self._track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._speeds: Dict[int, float] = {}

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, float]]:
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()
        frame_speeds: Dict[int, float] = {}

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)

            if not self._filter_class(cls):
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            history = self._track_history[tid]
            history.append((cx, cy))

            speed = self._compute_speed(history)
            self._speeds[tid] = speed
            frame_speeds[tid] = speed

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
            label = f"ID:{tid} {speed:.1f}"
            if self.pixels_per_meter != 1.0:
                label += " km/h"
            else:
                label += " px/f"
            cv2.putText(
                annotated, label,
                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
            )

        return annotated, frame_speeds

    def get_speeds(self) -> Dict[int, float]:
        """Return the latest speed for every tracked object."""
        return dict(self._speeds)

    def get_results(self) -> Dict[str, Any]:
        return {"speeds": self.get_speeds()}

    def reset(self):
        super().reset()
        self._track_history.clear()
        self._speeds.clear()

    def _compute_speed(self, history: deque) -> float:
        if len(history) < 2:
            return 0.0
        p0 = np.array(history[0])
        p1 = np.array(history[-1])
        dist_px = float(np.linalg.norm(p1 - p0))
        n_frames = len(history) - 1

        if self.pixels_per_meter == 1.0:
            return dist_px / n_frames

        dist_m = dist_px / self.pixels_per_meter
        time_s = n_frames / self.fps
        if time_s == 0:
            return 0.0
        speed_mps = dist_m / time_s
        return speed_mps * 3.6  # m/s -> km/h
