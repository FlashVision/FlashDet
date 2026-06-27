"""LiveInference — high-level wrapper for real-time webcam/video detection."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class LiveInference(BaseSolution):
    """Run real-time object detection on a webcam or video file.

    This is a convenience class that wraps camera capture, detection,
    visualisation and optional video recording into a single call.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Multi-object tracker.  Pass *None* to disable tracking.
    source : int | str
        Camera index (e.g. ``0``) or path to a video file.
    output_path : str | None
        If provided, the annotated stream is saved to this path.
    show_labels : bool
        Draw class labels next to boxes.
    show_confidence : bool
        Draw confidence scores.
    show_boxes : bool
        Draw bounding boxes.
    show_fps : bool
        Overlay the current FPS.
    class_names : list[str] | None
        Mapping from class index to human-readable name.
    window_name : str
        OpenCV window title when displaying.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        source: Union[int, str] = 0,
        output_path: Optional[str] = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        show_boxes: bool = True,
        show_fps: bool = True,
        class_names: Optional[List[str]] = None,
        window_name: str = "FlashDet Live",
    ):
        super().__init__(predictor, tracker)
        self.source = source
        self.output_path = output_path
        self.show_labels = show_labels
        self.show_confidence = show_confidence
        self.show_boxes = show_boxes
        self.show_fps = show_fps
        self.class_names = class_names
        self.window_name = window_name

        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._t_prev: float = 0.0
        self._running: bool = False

        rng = np.random.RandomState(42)
        self._palette = [
            tuple(int(c) for c in rng.randint(60, 255, 3))
            for _ in range(80)
        ]

    def run(self, max_frames: int = 0, on_frame: Optional[Callable] = None):
        """Start the live inference loop.

        Parameters
        ----------
        max_frames : int
            Stop after this many frames (0 = run until ``q`` or video ends).
        on_frame : callable | None
            Optional callback ``fn(frame, detections)`` called per frame.
        """
        self._open_source()
        self._running = True
        self._t_prev = time.perf_counter()

        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    break

                annotated, results = self.process_frame(frame)

                if on_frame is not None:
                    on_frame(annotated, results)

                if self._writer is not None:
                    self._writer.write(annotated)

                cv2.imshow(self.window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if max_frames > 0 and self._frame_count >= max_frames:
                    break
        finally:
            self.stop()

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_count += 1
        t_now = time.perf_counter()
        dt = t_now - self._t_prev
        self._fps = 1.0 / dt if dt > 0 else 0.0
        self._t_prev = t_now

        detections = self._detect(frame)
        annotated = frame.copy()

        result_list: List[Dict[str, Any]] = []

        if self.tracker is not None:
            tracks = self.tracker.update(detections)
            for trk in tracks:
                x1, y1, x2, y2, tid, score, cls = trk
                tid, cls = int(tid), int(cls)
                entry = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": round(float(score), 3),
                    "class_id": cls,
                    "track_id": tid,
                }
                result_list.append(entry)
                self._draw_detection(annotated, x1, y1, x2, y2, score, cls, tid)
        else:
            for det in detections:
                x1, y1, x2, y2, score, cls = det
                cls = int(cls)
                entry = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": round(float(score), 3),
                    "class_id": cls,
                    "track_id": None,
                }
                result_list.append(entry)
                self._draw_detection(annotated, x1, y1, x2, y2, score, cls)

        if self.show_fps:
            cv2.putText(
                annotated, f"FPS: {self._fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

        return annotated, {
            "detections": result_list,
            "fps": round(self._fps, 1),
            "frame_idx": self._frame_count,
        }

    def stop(self):
        """Release camera and writer resources."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        cv2.destroyAllWindows()

    def get_results(self) -> Dict[str, Any]:
        return {
            "frames_processed": self._frame_count,
            "current_fps": round(self._fps, 1),
        }

    def reset(self):
        super().reset()
        self._frame_count = 0
        self._fps = 0.0

    def _open_source(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        if self.output_path is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))

    def _draw_detection(
        self,
        img: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        score: float, cls: int,
        track_id: Optional[int] = None,
    ):
        color = self._palette[cls % len(self._palette)]

        if self.show_boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        parts: List[str] = []
        if track_id is not None:
            parts.append(f"ID:{track_id}")
        if self.show_labels:
            name = self.class_names[cls] if self.class_names and cls < len(self.class_names) else str(cls)
            parts.append(name)
        if self.show_confidence:
            parts.append(f"{score:.2f}")

        if parts:
            label = " ".join(parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                img,
                (int(x1), int(y1) - th - 6),
                (int(x1) + tw + 4, int(y1)),
                color, -1,
            )
            cv2.putText(
                img, label,
                (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
