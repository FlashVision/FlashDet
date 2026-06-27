"""TrafficFlow — direction-aware traffic flow analysis."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class TrafficFlow(BaseSolution):
    """Analyse traffic flow: direction, speed, and volume.

    Tracks are grouped by their dominant direction of travel.  The
    solution computes per-direction counts, average speed, and draws
    flow vectors on the annotated frame.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    n_directions : int
        Number of directional bins (8 = N/NE/E/SE/S/SW/W/NW).
    min_displacement : float
        Minimum total displacement (in pixels) for a track to be
        assigned a direction.
    history_length : int
        Past positions to keep per track.
    fps : float
        Video frame rate for speed estimation.
    pixels_per_meter : float
        Calibration factor (1.0 = report px/frame).
    classes : list[int] | None
        Only analyse these class IDs.
    """

    _DIRECTION_LABELS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    @staticmethod
    def _evenly_spaced_labels(n_directions: int) -> list:
        """Return *n_directions* evenly-spaced compass labels from the 8-point rose."""
        all_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        if n_directions >= 8:
            return list(all_labels)
        step = 8 // n_directions
        return [all_labels[i * step] for i in range(n_directions)]

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        n_directions: int = 8,
        min_displacement: float = 30.0,
        history_length: int = 30,
        fps: float = 30.0,
        pixels_per_meter: float = 1.0,
        classes: Optional[List[int]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.n_directions = n_directions
        self.min_displacement = min_displacement
        self.history_length = history_length
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter

        self._track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=history_length)
        )
        self._direction_counts: Dict[str, int] = defaultdict(int)
        self._direction_speeds: Dict[str, List[float]] = defaultdict(list)
        self._assigned_tracks: set = set()
        self._frame_idx: int = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_idx += 1
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)
            if not self._filter_class(cls):
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            self._track_history[tid].append((cx, cy))

            history = self._track_history[tid]
            direction, speed, angle = self._compute_flow(history)

            color = (0, 255, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            if direction is not None:
                # Register this track's direction once
                if tid not in self._assigned_tracks:
                    self._direction_counts[direction] += 1
                    self._assigned_tracks.add(tid)
                self._direction_speeds[direction].append(speed)

                # Draw flow arrow from centre
                arrow_len = 30
                ex = int(cx + arrow_len * math.cos(angle))
                ey = int(cy + arrow_len * math.sin(angle))
                cv2.arrowedLine(annotated, (int(cx), int(cy)), (ex, ey), (0, 200, 255), 2)
                cv2.putText(
                    annotated, f"{direction}",
                    (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1,
                )

        # Draw summary panel
        self._draw_summary(annotated)

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        summary: Dict[str, Dict[str, Any]] = {}
        for d_label in self._evenly_spaced_labels(self.n_directions):
            count = self._direction_counts.get(d_label, 0)
            speeds = self._direction_speeds.get(d_label, [])
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            summary[d_label] = {
                "count": count,
                "avg_speed": round(avg_speed, 2),
            }
        return {
            "frame_idx": self._frame_idx,
            "total_tracked": sum(self._direction_counts.values()),
            "directions": summary,
        }

    def reset(self):
        super().reset()
        self._track_history.clear()
        self._direction_counts.clear()
        self._direction_speeds.clear()
        self._assigned_tracks.clear()
        self._frame_idx = 0

    def _compute_flow(
        self, history: deque
    ) -> Tuple[Optional[str], float, float]:
        """Compute direction label, speed, and angle from track history."""
        if len(history) < 5:
            return None, 0.0, 0.0

        p0 = np.array(history[0])
        p1 = np.array(history[-1])
        displacement = p1 - p0
        dist = float(np.linalg.norm(displacement))

        if dist < self.min_displacement:
            return None, 0.0, 0.0

        angle = math.atan2(displacement[1], displacement[0])

        n_frames = len(history) - 1
        if self.pixels_per_meter == 1.0:
            speed = dist / n_frames
        else:
            dist_m = dist / self.pixels_per_meter
            time_s = n_frames / self.fps
            speed = (dist_m / time_s) * 3.6 if time_s > 0 else 0.0

        # Map angle to direction bin
        # angle=0 is East, pi/2 is South (OpenCV coords)
        # Shift so N (up, angle=-pi/2) maps to bin 0
        shifted = (angle + math.pi / 2) % (2 * math.pi)
        bin_size = 2 * math.pi / self.n_directions
        bin_idx = int((shifted + bin_size / 2) % (2 * math.pi) / bin_size)
        bin_idx = min(bin_idx, self.n_directions - 1)

        labels = self._evenly_spaced_labels(self.n_directions)
        direction = labels[bin_idx]

        return direction, speed, angle

    def _draw_summary(self, img: np.ndarray):
        y0 = 30
        cv2.putText(
            img, "Traffic Flow",
            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )
        y0 += 25
        for d_label in self._evenly_spaced_labels(self.n_directions):
            count = self._direction_counts.get(d_label, 0)
            if count > 0:
                cv2.putText(
                    img, f"{d_label}: {count}",
                    (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
                )
                y0 += 18
