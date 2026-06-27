"""Shared base class and utilities for all FlashDet solutions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np

from flashdet.trackers import FlashTracker


def run_detector(predictor, frame: np.ndarray) -> np.ndarray:
    """Run any predictor and normalize output to Nx6 array.

    Handles all predictor output formats:
    - np.ndarray (N, 6): [x1, y1, x2, y2, score, class_id]
    - List[Tuple[np.ndarray, float, int]]: FlashDet Predictor format
    - Object with .detections attribute
    - Dict with "boxes" key (raw model output)

    Returns:
        np.ndarray of shape (N, 6) with columns [x1, y1, x2, y2, score, class_id]
    """
    result = predictor(frame)

    if isinstance(result, np.ndarray) and result.ndim == 2 and result.shape[1] >= 6:
        return result[:, :6].astype(np.float64)

    if isinstance(result, list):
        if len(result) == 0:
            return np.empty((0, 6), dtype=np.float64)

        first = result[0]
        if isinstance(first, (tuple, list)) and len(first) == 3:
            dets = []
            for bbox, score, cls_id in result:
                if isinstance(bbox, np.ndarray):
                    row = np.array([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])
                else:
                    row = np.array([bbox[0], bbox[1], bbox[2], bbox[3], float(score), float(cls_id)])
                dets.append(row)
            return np.array(dets, dtype=np.float64) if dets else np.empty((0, 6), dtype=np.float64)

        if isinstance(first, dict) and "bbox" in first:
            dets = []
            for d in result:
                bbox = d["bbox"]
                score = d.get("confidence", d.get("score", 0.0))
                cls_id = d.get("class_id", 0)
                dets.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])
            return np.array(dets, dtype=np.float64) if dets else np.empty((0, 6), dtype=np.float64)

    if hasattr(result, "detections"):
        dets = np.asarray(result.detections, dtype=np.float64)
        if dets.ndim == 2 and dets.shape[1] >= 6:
            return dets[:, :6]
        return np.empty((0, 6), dtype=np.float64)

    if isinstance(result, dict) and "boxes" in result:
        boxes = np.asarray(result["boxes"])
        scores = np.asarray(result.get("scores", np.zeros(len(boxes))))
        classes = np.asarray(result.get("classes", np.zeros(len(boxes))))
        if len(boxes) > 0:
            return np.column_stack([boxes[:, :4], scores, classes]).astype(np.float64)

    return np.empty((0, 6), dtype=np.float64)


class BaseSolution(ABC):
    """Abstract base for all FlashDet solutions.

    Provides a shared interface and eliminates boilerplate:
    - Unified ``predictor`` + ``tracker`` handling
    - Shared ``_detect()`` that normalises any predictor output
    - Common ``process_frame`` / ``get_results`` / ``reset`` contract
    - Class filtering via ``classes`` parameter

    Parameters
    ----------
    predictor : object
        Any callable that accepts an image and returns detections.
    tracker : FlashTracker | None
        Multi-object tracker.  Defaults to ``FlashTracker()``
        unless the subclass explicitly passes *None*.
    classes : list[int] | None
        Only process these class IDs.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        classes: Optional[List[int]] = None,
    ):
        self.predictor = predictor
        self.tracker = tracker
        self.classes = classes

    def _ensure_tracker(self):
        """Lazily create a default tracker if none was provided."""
        if self.tracker is None:
            self.tracker = FlashTracker()
        return self.tracker

    def _detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detector and return normalised Nx6 array."""
        return run_detector(self.predictor, frame)

    def _filter_class(self, cls: int) -> bool:
        """Return True if *cls* should be processed."""
        return self.classes is None or cls in self.classes

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Process one video frame. Returns (annotated_frame, results)."""

    @abstractmethod
    def get_results(self) -> Any:
        """Return the latest results / statistics."""

    def reset(self):
        """Reset internal state. Override to clear solution-specific data."""
        if self.tracker is not None:
            self.tracker.reset()
