"""RegionCounter — count objects inside user-defined polygon regions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution

_Color = Tuple[int, int, int]


class RegionCounter(BaseSolution):
    """Count objects present inside one or more polygon regions.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    regions : dict[str, list[tuple[int, int]]]
        Mapping of region name -> list of polygon vertices.
    classes : list[int] | None
        Only count these class IDs.
    colors : dict[str, tuple[int,int,int]] | None
        Per-region BGR drawing colours.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        regions: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        classes: Optional[List[int]] = None,
        colors: Optional[Dict[str, _Color]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.regions: Dict[str, List[Tuple[int, int]]] = regions or {}

        if colors is not None:
            self._colors = colors
        else:
            palette: Sequence[_Color] = [
                (0, 255, 0), (255, 0, 0), (0, 0, 255),
                (255, 255, 0), (0, 255, 255), (255, 0, 255),
            ]
            self._colors = {
                name: palette[i % len(palette)]
                for i, name in enumerate(self.regions)
            }

        self._region_counts: Dict[str, int] = defaultdict(int)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        frame_counts: Dict[str, int] = {name: 0 for name in self.regions}
        annotated = frame.copy()

        for name, vertices in self.regions.items():
            color = self._colors.get(name, (0, 255, 0))
            pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)

            if not self._filter_class(cls):
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            drawn = False
            for name, vertices in self.regions.items():
                if self._point_in_polygon(cx, cy, vertices):
                    frame_counts[name] += 1
                    if not drawn:
                        color = self._colors.get(name, (0, 255, 0))
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(
                            annotated, f"ID:{tid}",
                            (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        )
                        drawn = True

        y_offset = 30
        for name, cnt in frame_counts.items():
            color = self._colors.get(name, (0, 255, 0))
            cv2.putText(
                annotated, f"{name}: {cnt}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            y_offset += 30

        self._region_counts = frame_counts
        return annotated, frame_counts

    def get_counts(self) -> Dict[str, int]:
        return dict(self._region_counts)

    def get_results(self) -> Dict[str, int]:
        return self.get_counts()

    def reset(self):
        super().reset()
        self._region_counts.clear()

    @staticmethod
    def _point_in_polygon(px: float, py: float, polygon: List[Tuple[int, int]]) -> bool:
        """Ray-casting point-in-polygon test."""
        n = len(polygon)
        inside = False
        x1, y1 = polygon[0]
        for i in range(1, n + 1):
            x2, y2 = polygon[i % n]
            if min(y1, y2) < py <= max(y1, y2):
                xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1
                if px <= xinters:
                    inside = not inside
            x1, y1 = x2, y2
        return inside
