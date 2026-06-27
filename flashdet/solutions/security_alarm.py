"""SecurityAlarm — trigger alerts when objects enter restricted zones."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class _IntrusionEvent:
    """Internal record of a single intrusion event."""

    __slots__ = ("track_id", "zone_id", "timestamp", "entry_frame",
                 "class_id", "position")

    def __init__(
        self, track_id: int, zone_id: int, entry_frame: int,
        class_id: int, position: Tuple[float, float],
    ):
        self.track_id = track_id
        self.zone_id = zone_id
        self.timestamp = time.time()
        self.entry_frame = entry_frame
        self.class_id = class_id
        self.position = position

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "zone_id": self.zone_id,
            "timestamp": self.timestamp,
            "entry_frame": self.entry_frame,
            "class_id": self.class_id,
            "position": self.position,
        }


class SecurityAlarm(BaseSolution):
    """Trigger alerts when tracked objects enter restricted zones.

    Restricted zones are defined as polygons.  When a track's centre
    enters a zone an intrusion event is logged.  A per-zone cooldown
    prevents repeated alerts for the same track lingering inside a zone.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    restricted_zones : list[np.ndarray] | None
        List of polygons (Nx2 int32 arrays).
    alert_cooldown : float
        Minimum seconds between alerts for the same (track, zone) pair.
    classes : list[int] | None
        Only raise alerts for these class IDs.
    max_log_size : int
        Maximum number of intrusion events to keep in the log.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        restricted_zones: Optional[List[np.ndarray]] = None,
        alert_cooldown: float = 5.0,
        classes: Optional[List[int]] = None,
        max_log_size: int = 1000,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.restricted_zones: List[np.ndarray] = self._normalize_zones(restricted_zones)
        self.alert_cooldown = alert_cooldown

        self._intrusion_log: deque = deque(maxlen=max_log_size)
        self._active_alerts: Dict[Tuple[int, int], _IntrusionEvent] = {}
        self._last_alert_time: Dict[Tuple[int, int], float] = defaultdict(float)
        self._frame_idx: int = 0
        self._alert_callback: Optional[Any] = None

    @staticmethod
    def _normalize_zones(zones):
        if zones is None:
            return []
        if isinstance(zones, dict):
            return [np.asarray(p, dtype=np.int32) for p in zones.values()]
        return [np.asarray(p, dtype=np.int32) if not isinstance(p, np.ndarray) else p.astype(np.int32) for p in zones]

    def add_zone(self, polygon: np.ndarray) -> int:
        """Add a restricted zone. Returns its index."""
        polygon = np.asarray(polygon, dtype=np.int32)
        self.restricted_zones.append(polygon)
        return len(self.restricted_zones) - 1

    def set_alert_callback(self, callback):
        """Register a callback ``fn(event_dict)`` invoked on each new alert."""
        self._alert_callback = callback

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_idx += 1
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()
        now = time.time()

        current_zone_tracks: Dict[Tuple[int, int], Tuple[float, float, int]] = {}

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)

            if not self._filter_class(cls):
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            for zidx, zone in enumerate(self.restricted_zones):
                if cv2.pointPolygonTest(
                    zone.astype(np.float32), (cx, cy), False
                ) >= 0:
                    key = (tid, zidx)
                    current_zone_tracks[key] = (cx, cy, cls)

                    if key not in self._active_alerts:
                        elapsed = now - self._last_alert_time[key]
                        if elapsed >= self.alert_cooldown:
                            event = _IntrusionEvent(
                                tid, zidx, self._frame_idx, cls, (cx, cy),
                            )
                            self._active_alerts[key] = event
                            self._intrusion_log.append(event)
                            self._last_alert_time[key] = now
                            if self._alert_callback is not None:
                                self._alert_callback(event.to_dict())

                    cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                    cv2.putText(
                        annotated, "ALERT",
                        (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )

        expired = [k for k in self._active_alerts if k not in current_zone_tracks]
        for k in expired:
            del self._active_alerts[k]

        for zidx, zone in enumerate(self.restricted_zones):
            overlay = annotated.copy()
            has_alert = any(k[1] == zidx for k in self._active_alerts)
            color = (0, 0, 255) if has_alert else (0, 140, 255)
            cv2.fillPoly(overlay, [zone], color)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
            cv2.polylines(annotated, [zone], True, color, 2)
            centroid = zone.mean(axis=0).astype(int)
            cv2.putText(
                annotated, f"RESTRICTED {zidx}",
                (centroid[0] - 40, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        n_active = len(self._active_alerts)
        status_color = (0, 0, 255) if n_active > 0 else (0, 200, 0)
        cv2.putText(
            annotated,
            f"Active alerts: {n_active}  |  Total intrusions: {len(self._intrusion_log)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2,
        )

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        return {
            "active_alerts": [e.to_dict() for e in self._active_alerts.values()],
            "intrusion_log": [e.to_dict() for e in self._intrusion_log],
            "total_intrusions": len(self._intrusion_log),
        }

    def reset(self):
        super().reset()
        self._active_alerts.clear()
        self._intrusion_log.clear()
        self._last_alert_time.clear()
        self._frame_idx = 0
