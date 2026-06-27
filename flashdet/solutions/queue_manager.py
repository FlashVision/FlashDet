"""QueueManager — monitor queues and count people waiting in defined regions."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class QueueManager(BaseSolution):
    """Monitor queue regions and count people waiting in each zone.

    Define one or more queue regions as polygons.  The manager tracks how
    many people are inside each region, how long they stay, and reports
    peak occupancy.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    queue_regions : list[np.ndarray] | None
        List of polygons, each an Nx2 int32 array of vertices.
    classes : list[int] | None
        Only count objects whose class ID is in this list.
    fps : float
        Video frame rate, used for wait-time estimation.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        queue_regions: Optional[List[np.ndarray]] = None,
        classes: Optional[List[int]] = None,
        fps: float = 30.0,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.queue_regions = self._normalize_regions(queue_regions)
        self.fps = fps

        self._n_regions: int = 0
        self._track_enter_frame: Dict[int, Dict[int, int]] = defaultdict(dict)
        self._current_ids: Dict[int, set] = defaultdict(set)
        self._peak_counts: Dict[int, int] = defaultdict(int)
        self._wait_times: Dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
        self._frame_idx: int = 0

    @staticmethod
    def _normalize_regions(regions):
        if regions is None:
            return None
        if isinstance(regions, dict):
            return [np.asarray(p, dtype=np.int32) for p in regions.values()]
        return [np.array(p, dtype=np.int32) if not isinstance(p, np.ndarray) else p.astype(np.int32) for p in regions]

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.queue_regions is None:
            h, w = frame.shape[:2]
            self.queue_regions = [
                np.array(
                    [[w // 3, h // 3], [2 * w // 3, h // 3],
                     [2 * w // 3, 2 * h // 3], [w // 3, 2 * h // 3]],
                    dtype=np.int32,
                )
            ]
        self._n_regions = len(self.queue_regions)
        self._frame_idx += 1

        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()
        region_ids: Dict[int, set] = defaultdict(set)

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)
            if not self._filter_class(cls):
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            for ridx, poly in enumerate(self.queue_regions):
                if cv2.pointPolygonTest(poly.astype(np.float32), (cx, cy), False) >= 0:
                    region_ids[ridx].add(tid)
                    if tid not in self._track_enter_frame[ridx]:
                        self._track_enter_frame[ridx][tid] = self._frame_idx

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"ID:{tid}",
                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        for ridx in range(self._n_regions):
            current = region_ids.get(ridx, set())
            left = self._current_ids[ridx] - current
            for tid in left:
                enter_f = self._track_enter_frame[ridx].pop(tid, self._frame_idx)
                wait_frames = self._frame_idx - enter_f
                self._wait_times[ridx].append(wait_frames / self.fps)
            self._current_ids[ridx] = current
            count = len(current)
            if count > self._peak_counts[ridx]:
                self._peak_counts[ridx] = count

        for ridx, poly in enumerate(self.queue_regions):
            cv2.polylines(annotated, [poly], True, (0, 255, 255), 2)
            count = len(self._current_ids.get(ridx, set()))
            centroid = poly.mean(axis=0).astype(int)
            cv2.putText(
                annotated, f"Q{ridx}: {count}",
                (centroid[0] - 20, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        regions: List[Dict[str, Any]] = []
        total = 0
        for ridx in range(self._n_regions):
            length = len(self._current_ids.get(ridx, set()))
            total += length
            waits = self._wait_times[ridx]
            avg_wait = float(np.mean(waits)) if waits else 0.0
            regions.append({
                "queue_length": length,
                "avg_wait_time": round(avg_wait, 2),
                "peak_count": self._peak_counts.get(ridx, 0),
            })
        return {"regions": regions, "total_waiting": total}

    def reset(self):
        super().reset()
        self._track_enter_frame.clear()
        self._current_ids.clear()
        self._peak_counts.clear()
        self._wait_times.clear()
        self._frame_idx = 0
