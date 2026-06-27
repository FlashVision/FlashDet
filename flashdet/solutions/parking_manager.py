"""ParkingManager — track parking spot occupancy using polygon regions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class ParkingManager(BaseSolution):
    """Define parking spots as polygons and track their occupancy.

    Each parking spot is defined by a polygon (Nx2 array of vertices).
    A spot is considered occupied when a detection's bounding-box centre
    falls inside the polygon.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.
    parking_spots : list[np.ndarray] | None
        List of polygons, each an Nx2 int32 array.
    occupancy_threshold : float
        Minimum IoU overlap between detection bbox and spot polygon for
        the spot to be considered occupied.  When set to 0 (default),
        only the centre-point test is used.
    classes : list[int] | None
        Only consider these class IDs (e.g. ``[2]`` for cars in COCO).
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        parking_spots: Optional[List[np.ndarray]] = None,
        occupancy_threshold: float = 0.0,
        classes: Optional[List[int]] = None,
    ):
        super().__init__(predictor, tracker, classes)
        self._ensure_tracker()
        self.parking_spots: List[np.ndarray] = parking_spots or []
        self.occupancy_threshold = occupancy_threshold

        self._spot_status: Dict[int, bool] = {}
        self._spot_track_ids: Dict[int, Optional[int]] = {}

    def add_spot(self, polygon: np.ndarray, spot_id: Optional[int] = None) -> int:
        """Add a parking spot polygon and return its index."""
        polygon = np.asarray(polygon, dtype=np.int32)
        idx = spot_id if spot_id is not None else len(self.parking_spots)
        if idx < len(self.parking_spots):
            self.parking_spots[idx] = polygon
        else:
            self.parking_spots.append(polygon)
            idx = len(self.parking_spots) - 1
        return idx

    def load_spots(self, path: Union[str, Path]):
        """Load parking spot definitions from a JSON file."""
        data = json.loads(Path(path).read_text())
        self.parking_spots.clear()
        for entry in data["spots"]:
            poly = np.array(entry["polygon"], dtype=np.int32)
            self.parking_spots.append(poly)

    def save_spots(self, path: Union[str, Path]):
        """Save current spot definitions to a JSON file."""
        spots = []
        for idx, poly in enumerate(self.parking_spots):
            spots.append({"id": idx, "polygon": poly.tolist()})
        Path(path).write_text(json.dumps({"spots": spots}, indent=2))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        detections = self._detect(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()

        filtered_tracks = []
        for trk in tracks:
            cls = int(trk[6])
            if self._filter_class(cls):
                filtered_tracks.append(trk)

        self._update_occupancy(filtered_tracks)

        for sidx, poly in enumerate(self.parking_spots):
            occupied = self._spot_status.get(sidx, False)
            color = (0, 0, 200) if occupied else (0, 200, 0)
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
            cv2.polylines(annotated, [poly], True, color, 2)
            centroid = poly.mean(axis=0).astype(int)
            label = "X" if occupied else "O"
            cv2.putText(
                annotated, label,
                (centroid[0] - 5, centroid[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        for trk in filtered_tracks:
            x1, y1, x2, y2 = trk[:4]
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)

        total = len(self.parking_spots)
        occupied_count = sum(1 for v in self._spot_status.values() if v)
        cv2.putText(
            annotated,
            f"Available: {total - occupied_count}/{total}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        total = len(self.parking_spots)
        occupied = sum(1 for v in self._spot_status.values() if v)
        available = total - occupied
        rate = (occupied / total * 100) if total > 0 else 0.0
        spots_detail = []
        for sidx in range(total):
            spots_detail.append({
                "spot_id": sidx,
                "occupied": self._spot_status.get(sidx, False),
                "track_id": self._spot_track_ids.get(sidx),
            })
        return {
            "total_spots": total,
            "occupied": occupied,
            "available": available,
            "occupancy_rate": round(rate, 1),
            "spots": spots_detail,
        }

    def reset(self):
        super().reset()
        self._spot_status.clear()
        self._spot_track_ids.clear()

    def _update_occupancy(self, tracks: list):
        self._spot_status = {i: False for i in range(len(self.parking_spots))}
        self._spot_track_ids = {i: None for i in range(len(self.parking_spots))}

        for trk in tracks:
            x1, y1, x2, y2, tid = trk[0], trk[1], trk[2], trk[3], int(trk[4])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if self.occupancy_threshold > 0:
                for sidx, poly in enumerate(self.parking_spots):
                    if self._bbox_polygon_iou(trk[:4], poly) >= self.occupancy_threshold:
                        self._spot_status[sidx] = True
                        self._spot_track_ids[sidx] = tid
            else:
                for sidx, poly in enumerate(self.parking_spots):
                    if cv2.pointPolygonTest(
                        poly.astype(np.float32), (cx, cy), False
                    ) >= 0:
                        self._spot_status[sidx] = True
                        self._spot_track_ids[sidx] = tid

    @staticmethod
    def _bbox_polygon_iou(bbox: np.ndarray, polygon: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox[:4].astype(int)
        poly_pts = polygon.astype(int)

        all_x = np.concatenate([poly_pts[:, 0], [x1, x2]])
        all_y = np.concatenate([poly_pts[:, 1], [y1, y2]])
        ox, oy = all_x.min(), all_y.min()
        w = all_x.max() - ox + 1
        h = all_y.max() - oy + 1

        mask_bbox = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_bbox, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), 1, -1)

        mask_poly = np.zeros((h, w), dtype=np.uint8)
        shifted = poly_pts - np.array([[ox, oy]])
        cv2.fillPoly(mask_poly, [shifted], 1)

        inter = int((mask_bbox & mask_poly).sum())
        union = int((mask_bbox | mask_poly).sum())
        return inter / union if union > 0 else 0.0
