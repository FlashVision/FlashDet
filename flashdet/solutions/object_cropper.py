"""ObjectCropper — crop and save detected objects from video frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class ObjectCropper(BaseSolution):
    """Crop detected objects and optionally save them to disk.

    Each detection (or tracked object) is cropped from the frame.
    Crops can be saved as individual images, collected in-memory,
    or both.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.  When used, crops are named by track ID,
        preventing duplicate saves of the same object.
    output_dir : str | None
        Directory to save crops.  ``None`` = don't save (in-memory only).
    save_format : str
        Image format for saved crops (``"jpg"`` or ``"png"``).
    padding : float
        Fractional padding around the bounding box (0.1 = 10% each side).
    min_size : int
        Minimum crop width or height in pixels (smaller crops skipped).
    classes : list[int] | None
        Only crop these class IDs.
    one_per_track : bool
        If True and tracker is used, save only the first crop per track ID.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        output_dir: Optional[str] = None,
        save_format: str = "jpg",
        padding: float = 0.0,
        min_size: int = 10,
        classes: Optional[List[int]] = None,
        one_per_track: bool = False,
    ):
        super().__init__(predictor, tracker, classes)
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_format = save_format
        self.padding = padding
        self.min_size = min_size
        self.one_per_track = one_per_track

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._saved_track_ids: set = set()
        self._crop_count: int = 0
        self._frame_idx: int = 0
        self._last_crops: List[np.ndarray] = []

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_idx += 1
        detections = self._detect(frame)

        use_tracker = self.tracker is not None
        if use_tracker:
            data = self.tracker.update(detections)
        else:
            data = detections

        annotated = frame.copy()
        h_img, w_img = frame.shape[:2]
        self._last_crops = []

        for idx, row in enumerate(data):
            if use_tracker:
                x1, y1, x2, y2, tid, score, cls = row
                tid, cls = int(tid), int(cls)
            else:
                x1, y1, x2, y2, score, cls = row
                cls = int(cls)
                tid = None

            if not self._filter_class(cls):
                continue

            if self.one_per_track and tid is not None and tid in self._saved_track_ids:
                continue

            pw = (x2 - x1) * self.padding
            ph = (y2 - y1) * self.padding
            cx1 = max(0, int(x1 - pw))
            cy1 = max(0, int(y1 - ph))
            cx2 = min(w_img, int(x2 + pw))
            cy2 = min(h_img, int(y2 + ph))

            if (cx2 - cx1) < self.min_size or (cy2 - cy1) < self.min_size:
                continue

            crop = frame[cy1:cy2, cx1:cx2].copy()
            self._last_crops.append(crop)
            self._crop_count += 1

            if self.output_dir is not None:
                if tid is not None:
                    fname = f"f{self._frame_idx:06d}_id{tid}_cls{cls}.{self.save_format}"
                else:
                    fname = f"f{self._frame_idx:06d}_d{idx}_cls{cls}.{self.save_format}"
                cv2.imwrite(str(self.output_dir / fname), crop)

            if tid is not None:
                self._saved_track_ids.add(tid)

            # Draw crop region
            cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (255, 200, 0), 2)

        return annotated, self.get_results()

    def get_crops(self) -> List[np.ndarray]:
        """Return the crops from the most recent frame."""
        return list(self._last_crops)

    def get_results(self) -> Dict[str, Any]:
        return {
            "frame_idx": self._frame_idx,
            "total_crops": self._crop_count,
            "last_frame_crops": len(self._last_crops),
        }

    def reset(self):
        super().reset()
        self._saved_track_ids.clear()
        self._crop_count = 0
        self._frame_idx = 0
        self._last_crops = []
