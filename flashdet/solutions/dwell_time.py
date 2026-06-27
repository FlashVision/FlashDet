"""DwellTimeAnalyzer — measure how long objects stay in defined zones."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class DwellTimeAnalyzer(BaseSolution):
    """Measure how long each tracked object spends inside defined zones.

    Unlike ``QueueManager`` (which focuses on queue length / wait time),
    this solution focuses on *per-object* dwell duration for any purpose
    — retail analytics, suspicious loitering detection, etc.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    zones : dict[str, list[tuple[int, int]]]
        Named zones as polygon vertex lists.
    fps : float
        Video frame rate (for converting frame counts to seconds).
    alert_threshold_s : float
        If a track dwells longer than this (seconds), it is flagged.
    classes : list[int] | None
        Only analyse these class IDs.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        zones: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        fps: float = 30.0,
        alert_threshold_s: float = 30.0,
        classes: Optional[List[int]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.zones: Dict[str, List[Tuple[int, int]]] = zones or {}
        self.fps = fps
        self.alert_threshold_s = alert_threshold_s

        # {zone_name: {track_id: frames_inside}}
        self._dwell_frames: Dict[str, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # completed dwell records: (zone, track_id, duration_s)
        self._completed: List[Dict[str, Any]] = []
        self._active_in_zone: Dict[str, set] = defaultdict(set)
        self._frame_idx: int = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_idx += 1
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()

        # Draw zones
        for name, vertices in self.zones.items():
            pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (200, 200, 0), 2)
            centroid = pts.mean(axis=0).astype(int).flatten()
            cv2.putText(
                annotated, name,
                (centroid[0] - 20, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2,
            )

        current_zone_tracks: Dict[str, set] = defaultdict(set)

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)
            if not self._filter_class(cls):
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            for name, vertices in self.zones.items():
                poly = np.array(vertices, dtype=np.float32)
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    self._dwell_frames[name][tid] += 1
                    current_zone_tracks[name].add(tid)

            # Draw track
            dwell_s = self._max_dwell_seconds(tid)
            is_alert = dwell_s >= self.alert_threshold_s
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                annotated, f"ID:{tid} {dwell_s:.1f}s",
                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        # Detect tracks that left a zone
        for name in self.zones:
            prev = self._active_in_zone[name]
            curr = current_zone_tracks[name]
            left = prev - curr
            for tid in left:
                frames = self._dwell_frames[name].pop(tid, 0)
                if frames > 0:
                    self._completed.append({
                        "zone": name,
                        "track_id": tid,
                        "duration_s": round(frames / self.fps, 2),
                    })
            self._active_in_zone[name] = curr

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        active: Dict[str, List[Dict[str, Any]]] = {}
        for name in self.zones:
            entries = []
            for tid, frames in self._dwell_frames[name].items():
                dur = round(frames / self.fps, 2)
                entries.append({
                    "track_id": tid,
                    "duration_s": dur,
                    "alert": dur >= self.alert_threshold_s,
                })
            active[name] = entries
        return {
            "frame_idx": self._frame_idx,
            "active_dwells": active,
            "completed_dwells": self._completed[-50:],
        }

    def reset(self):
        super().reset()
        self._dwell_frames.clear()
        self._completed.clear()
        self._active_in_zone.clear()
        self._frame_idx = 0

    def _max_dwell_seconds(self, tid: int) -> float:
        max_frames = 0
        for zone_data in self._dwell_frames.values():
            max_frames = max(max_frames, zone_data.get(tid, 0))
        return max_frames / self.fps
