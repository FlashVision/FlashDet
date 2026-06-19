"""WorkoutMonitor — count exercise repetitions from bounding-box motion."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers.byte_tracker import ByteTracker


class WorkoutMonitor:
    """Count exercise repetitions by analysing vertical bounding-box movement.

    For exercises like squats, push-ups, or jumping jacks the bounding box
    height or vertical centre oscillates periodically.  The monitor detects
    these oscillations and counts full up-down cycles as repetitions.

    The state machine:

    *  ``up``   → the person's bbox centre is above the midpoint threshold.
    *  ``down`` → the person's bbox centre is below the midpoint threshold.

    A rep is counted on each ``down → up`` transition (configurable).

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : ByteTracker | None
        Multi-object tracker.
    exercise_type : str
        Hint label such as ``"squats"``, ``"pushups"``, ``"jumping_jacks"``.
    up_threshold : float
        Fractional bbox-height position that counts as "up" (0 = top).
        Default 0.4 — the centre is in the upper 40 % of the bbox's
        recent range.
    down_threshold : float
        Fractional position that counts as "down".  Default 0.6.
    window : int
        Number of frames over which to compute the y-range for
        normalisation.
    classes : list[int] | None
        Only monitor these class IDs (typically ``[0]`` for *person*).
    """

    _STATE_UP = "up"
    _STATE_DOWN = "down"
    _STATE_INIT = "init"

    def __init__(
        self,
        predictor,
        tracker: Optional[ByteTracker] = None,
        exercise_type: str = "squats",
        up_threshold: float = 0.4,
        down_threshold: float = 0.6,
        window: int = 60,
        classes: Optional[List[int]] = None,
    ):
        self.predictor = predictor
        self.tracker = tracker or ByteTracker()
        self.exercise_type = exercise_type
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.window = window
        self.classes = classes

        self._cy_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._bh_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window)
        )
        self._states: Dict[int, str] = defaultdict(lambda: self._STATE_INIT)
        self._rep_counts: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process one frame and update rep counts.

        Returns
        -------
        annotated : np.ndarray
            Frame with bounding boxes, states and rep counts.
        results : dict
            ``{"tracks": {id: {"rep_count": …, "state": …, "exercise": …}}}``
        """
        detections = self._run_detector(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)
            if self.classes is not None and cls not in self.classes:
                continue

            cy = (y1 + y2) / 2.0
            bh = y2 - y1

            self._cy_history[tid].append(cy)
            self._bh_history[tid].append(bh)

            frac = self._normalised_position(tid, cy)
            prev_state = self._states[tid]

            if frac is not None:
                if frac <= self.up_threshold:
                    self._states[tid] = self._STATE_UP
                elif frac >= self.down_threshold:
                    self._states[tid] = self._STATE_DOWN

                if prev_state == self._STATE_DOWN and self._states[tid] == self._STATE_UP:
                    self._rep_counts[tid] += 1

            # Draw
            color = self._state_color(self._states[tid])
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            state_label = self._states[tid]
            reps = self._rep_counts[tid]
            label = f"ID:{tid} {self.exercise_type} R:{reps} [{state_label}]"
            cv2.putText(
                annotated, label,
                (int(x1), int(y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

            if frac is not None:
                bar_h = int((y2 - y1) * frac)
                bar_x = int(x2) + 5
                cv2.rectangle(
                    annotated,
                    (bar_x, int(y1)), (bar_x + 8, int(y1) + bar_h),
                    color, -1,
                )

        total_reps = sum(self._rep_counts.values())
        cv2.putText(
            annotated,
            f"{self.exercise_type.upper()} — Total reps: {total_reps}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2,
        )

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Return per-track rep counts and states."""
        tracks: Dict[int, Dict[str, Any]] = {}
        for tid in self._rep_counts:
            tracks[tid] = {
                "rep_count": self._rep_counts[tid],
                "current_state": self._states[tid],
                "exercise_type": self.exercise_type,
            }
        return {"tracks": tracks, "total_reps": sum(self._rep_counts.values())}

    def reset(self):
        """Clear all rep counts and history."""
        self._cy_history.clear()
        self._bh_history.clear()
        self._states.clear()
        self._rep_counts.clear()
        self.tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalised_position(self, tid: int, cy: float) -> Optional[float]:
        """Map *cy* to [0, 1] using the observed min/max range for *tid*."""
        history = self._cy_history[tid]
        if len(history) < 10:
            return None
        y_min = min(history)
        y_max = max(history)
        span = y_max - y_min
        if span < 5:
            return None
        return (cy - y_min) / span

    @staticmethod
    def _state_color(state: str) -> Tuple[int, int, int]:
        if state == WorkoutMonitor._STATE_UP:
            return (0, 255, 0)
        if state == WorkoutMonitor._STATE_DOWN:
            return (0, 0, 255)
        return (200, 200, 200)

    def _run_detector(self, frame: np.ndarray) -> np.ndarray:
        result = self.predictor(frame)
        if isinstance(result, np.ndarray):
            return result
        if hasattr(result, "detections"):
            return np.asarray(result.detections, dtype=np.float64)
        return np.empty((0, 6), dtype=np.float64)
