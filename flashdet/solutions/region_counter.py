"""RegionCounter — count objects inside user-defined polygon regions."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from flashdet.trackers.byte_tracker import ByteTracker

_Color = Tuple[int, int, int]


class RegionCounter:
    """Count objects present inside one or more polygon regions.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : ByteTracker | None
        Multi-object tracker.
    regions : dict[str, list[tuple[int, int]]]
        Mapping of region name → list of polygon vertices.
        Example: ``{"zone_A": [(100,100), (400,100), (400,400), (100,400)]}``.
    classes : list[int] | None
        Only count these class IDs.
    colors : dict[str, tuple[int,int,int]] | None
        Per-region BGR drawing colours.  Auto-generated when *None*.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[ByteTracker] = None,
        regions: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        classes: Optional[List[int]] = None,
        colors: Optional[Dict[str, _Color]] = None,
    ):
        self.predictor = predictor
        self.tracker = tracker or ByteTracker()
        self.regions: Dict[str, List[Tuple[int, int]]] = regions or {}
        self.classes = classes

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """Process one frame.

        Returns
        -------
        annotated : np.ndarray
            Frame with region polygons, bounding boxes and per-region counts.
        counts : dict[str, int]
            ``{region_name: current_object_count}`` for this frame.
        """
        detections = self._run_detector(frame)
        tracks = self.tracker.update(detections)

        frame_counts: Dict[str, int] = {name: 0 for name in self.regions}
        annotated = frame.copy()

        # Draw regions
        for name, vertices in self.regions.items():
            color = self._colors.get(name, (0, 255, 0))
            pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        # Check each track
        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)

            if self.classes is not None and cls not in self.classes:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            for name, vertices in self.regions.items():
                if self._point_in_polygon(cx, cy, vertices):
                    frame_counts[name] += 1
                    color = self._colors.get(name, (0, 255, 0))
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(
                        annotated, f"ID:{tid}",
                        (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )
                    break  # each track counted only in its first matching region

        # Overlay counts
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
        """Return the most recent per-region counts."""
        return dict(self._region_counts)

    def reset(self):
        """Reset counter state."""
        self._region_counts.clear()
        self.tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_detector(self, frame: np.ndarray) -> np.ndarray:
        result = self.predictor(frame)
        if isinstance(result, np.ndarray):
            return result
        if hasattr(result, "detections"):
            return np.asarray(result.detections, dtype=np.float64)
        return np.empty((0, 6), dtype=np.float64)

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
