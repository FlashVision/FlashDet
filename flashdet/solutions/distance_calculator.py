"""DistanceCalculator — compute real-world distances between detected objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers.byte_tracker import ByteTracker


class DistanceCalculator:
    """Calculate real-world distance between detected objects.

    Supports two calibration modes:

    1. **pixels_per_meter** — simple flat-ground assumption where a constant
       factor converts pixel distance to metres.
    2. **perspective_transform** — four calibration points map image pixels
       to a bird's-eye-view plane, enabling accurate measurements even with
       perspective distortion.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : ByteTracker | None
        Multi-object tracker.  Defaults to a fresh ``ByteTracker()``.
    pixels_per_meter : float
        Simple calibration ratio.  Ignored when *src_points* / *dst_points*
        are provided for perspective calibration.
    src_points : np.ndarray | None
        4×2 array of calibration points in the image (pixel coords).
    dst_points : np.ndarray | None
        4×2 array of corresponding real-world positions (in metres).
    classes : list[int] | None
        Only compute distances for these class IDs.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[ByteTracker] = None,
        pixels_per_meter: float = 1.0,
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        classes: Optional[List[int]] = None,
    ):
        self.predictor = predictor
        self.tracker = tracker or ByteTracker()
        self.pixels_per_meter = pixels_per_meter
        self.classes = classes

        self._transform_mat: Optional[np.ndarray] = None
        if src_points is not None and dst_points is not None:
            src = np.asarray(src_points, dtype=np.float32).reshape(4, 2)
            dst = np.asarray(dst_points, dtype=np.float32).reshape(4, 2)
            self._transform_mat = cv2.getPerspectiveTransform(src, dst)

        self._last_distances: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._last_ids: List[int] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process one frame and compute pairwise distances.

        Returns
        -------
        annotated : np.ndarray
            Frame with bounding boxes and distance lines drawn.
        results : dict
            ``{"ids": […], "distance_matrix": 2-D list,
            "pairs": [{"id_a": …, "id_b": …, "distance_m": …}, …]}``
        """
        detections = self._run_detector(frame)
        tracks = self.tracker.update(detections)

        annotated = frame.copy()
        centres_px: List[Tuple[float, float]] = []
        track_ids: List[int] = []

        for trk in tracks:
            x1, y1, x2, y2, tid, score, cls = trk
            tid, cls = int(tid), int(cls)
            if self.classes is not None and cls not in self.classes:
                continue

            cx = (x1 + x2) / 2.0
            foot_y = y2
            centres_px.append((cx, foot_y))
            track_ids.append(tid)

            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"ID:{tid}",
                (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        n = len(centres_px)
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        real_points = self._to_real_world(centres_px)

        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(
                    np.array(real_points[i]) - np.array(real_points[j])
                ))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Draw top-k closest pairs
        pairs = self._sorted_pairs(track_ids, dist_matrix)
        for pair in pairs[:10]:
            ia = track_ids.index(pair["id_a"])
            ib = track_ids.index(pair["id_b"])
            p1 = (int(centres_px[ia][0]), int(centres_px[ia][1]))
            p2 = (int(centres_px[ib][0]), int(centres_px[ib][1]))
            color = (0, 0, 255) if pair["distance_m"] < 2.0 else (0, 200, 200)
            cv2.line(annotated, p1, p2, color, 1)
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                annotated, f"{pair['distance_m']:.1f}m",
                mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        self._last_distances = dist_matrix
        self._last_ids = track_ids

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Return the latest distance matrix and pair list."""
        pairs = self._sorted_pairs(self._last_ids, self._last_distances)
        return {
            "ids": list(self._last_ids),
            "distance_matrix": self._last_distances.tolist(),
            "pairs": pairs,
        }

    def calibrate(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ):
        """Set or update the perspective calibration.

        Parameters
        ----------
        src_points : np.ndarray
            4×2 image-space calibration points.
        dst_points : np.ndarray
            4×2 real-world calibration points (metres).
        """
        src = np.asarray(src_points, dtype=np.float32).reshape(4, 2)
        dst = np.asarray(dst_points, dtype=np.float32).reshape(4, 2)
        self._transform_mat = cv2.getPerspectiveTransform(src, dst)

    def reset(self):
        """Clear cached results."""
        self._last_distances = np.empty((0, 0), dtype=np.float64)
        self._last_ids = []
        self.tracker.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_real_world(
        self, points_px: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Convert pixel coordinates to real-world coordinates."""
        if not points_px:
            return []

        if self._transform_mat is not None:
            pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(pts, self._transform_mat)
            return [(float(p[0][0]), float(p[0][1])) for p in transformed]

        return [
            (px / self.pixels_per_meter, py / self.pixels_per_meter)
            for px, py in points_px
        ]

    @staticmethod
    def _sorted_pairs(
        ids: List[int], dist_matrix: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Build a sorted list of pairwise distances."""
        n = len(ids)
        pairs: List[Dict[str, Any]] = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append({
                    "id_a": ids[i],
                    "id_b": ids[j],
                    "distance_m": round(float(dist_matrix[i, j]), 3),
                })
        pairs.sort(key=lambda p: p["distance_m"])
        return pairs

    def _run_detector(self, frame: np.ndarray) -> np.ndarray:
        result = self.predictor(frame)
        if isinstance(result, np.ndarray):
            return result
        if hasattr(result, "detections"):
            return np.asarray(result.detections, dtype=np.float64)
        return np.empty((0, 6), dtype=np.float64)
