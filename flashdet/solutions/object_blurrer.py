"""ObjectBlurrer — blur or anonymize detected objects in video frames."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class ObjectBlurrer(BaseSolution):
    """Blur detected objects for privacy / anonymization.

    Applies a Gaussian or pixelation blur to bounding boxes of selected
    classes (e.g. faces, license plates).  Useful for GDPR compliance
    and privacy-preserving video processing.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker (optional, raw detections are used if None).
    blur_type : str
        ``"gaussian"`` for Gaussian blur, ``"pixelate"`` for mosaic effect.
    blur_strength : int
        Kernel size for Gaussian blur or block size for pixelation.
        Must be odd for Gaussian.
    classes : list[int] | None
        Only blur these class IDs.
    padding : float
        Fractional padding around the bounding box (0.1 = 10% each side).
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        blur_type: str = "gaussian",
        blur_strength: int = 51,
        classes: Optional[List[int]] = None,
        padding: float = 0.0,
    ):
        super().__init__(predictor, tracker, classes)
        self.blur_type = blur_type
        self.blur_strength = blur_strength | 1  # ensure odd
        self.padding = padding
        self._blur_count: int = 0
        self._frame_idx: int = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_idx += 1
        detections = self._detect(frame)

        if self.tracker is not None:
            data = self.tracker.update(detections)
            cls_col = 6
        else:
            data = detections
            cls_col = 5

        annotated = frame.copy()
        count = 0

        h_img, w_img = annotated.shape[:2]

        for row in data:
            cls = int(row[cls_col])
            if not self._filter_class(cls):
                continue

            x1, y1, x2, y2 = row[:4]
            pw = (x2 - x1) * self.padding
            ph = (y2 - y1) * self.padding
            x1 = max(0, int(x1 - pw))
            y1 = max(0, int(y1 - ph))
            x2 = min(w_img, int(x2 + pw))
            y2 = min(h_img, int(y2 + ph))

            if x2 <= x1 or y2 <= y1:
                continue

            roi = annotated[y1:y2, x1:x2]

            if self.blur_type == "pixelate":
                bw = max(2, self.blur_strength)
                small = cv2.resize(roi, (bw, bw), interpolation=cv2.INTER_LINEAR)
                annotated[y1:y2, x1:x2] = cv2.resize(
                    small, (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                annotated[y1:y2, x1:x2] = cv2.GaussianBlur(
                    roi, (self.blur_strength, self.blur_strength), 0,
                )

            count += 1

        self._blur_count += count
        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        return {
            "frame_idx": self._frame_idx,
            "total_blurred": self._blur_count,
        }

    def reset(self):
        super().reset()
        self._blur_count = 0
        self._frame_idx = 0
