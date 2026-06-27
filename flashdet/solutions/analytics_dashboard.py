"""AnalyticsDashboard — aggregate detection statistics over time."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from flashdet.trackers import FlashTracker
from flashdet.solutions._base import BaseSolution


class AnalyticsDashboard(BaseSolution):
    """Aggregate per-frame detection statistics and generate reports.

    Tracks detections per frame, per-class distributions, confidence
    histograms and time-windowed averages.

    Parameters
    ----------
    predictor : object
        FlashDet predictor returning Nx6 detections.
    tracker : FlashTracker | None
        Optional tracker — when provided, track-based statistics
        (unique IDs, active tracks) are also collected.
    window_size : int
        Number of recent frames used for windowed statistics.
    class_names : list[str] | None
        Human-readable class names indexed by class ID.
    conf_bins : int
        Number of bins in the confidence histogram.
    """

    def __init__(
        self,
        predictor,
        tracker: Optional[FlashTracker] = None,
        window_size: int = 300,
        class_names: Optional[List[str]] = None,
        conf_bins: int = 20,
    ):
        super().__init__(predictor, tracker)
        self.window_size = window_size
        self.class_names = class_names
        self.conf_bins = conf_bins

        self._frame_idx: int = 0
        self._det_counts: deque = deque(maxlen=window_size)
        self._class_counts: Dict[int, int] = defaultdict(int)
        self._window_class_counts: deque = deque(maxlen=window_size)
        self._conf_values: deque = deque(maxlen=window_size * 50)
        self._unique_track_ids: set = set()
        self._active_track_ids: deque = deque(maxlen=window_size)
        self._total_detections: int = 0

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._frame_idx += 1
        detections = self._detect(frame)

        frame_class_counts: Dict[int, int] = defaultdict(int)
        for det in detections:
            cls = int(det[5])
            frame_class_counts[cls] += 1
            self._class_counts[cls] += 1
            self._conf_values.append(float(det[4]))

        n_dets = len(detections)
        self._det_counts.append(n_dets)
        self._total_detections += n_dets
        self._window_class_counts.append(dict(frame_class_counts))

        if self.tracker is not None:
            tracks = self.tracker.update(detections)
            active_ids = set()
            for trk in tracks:
                tid = int(trk[4])
                active_ids.add(tid)
                self._unique_track_ids.add(tid)
            self._active_track_ids.append(len(active_ids))

        annotated = frame.copy()
        self._draw_overlay(annotated)

        return annotated, self.get_results()

    def get_results(self) -> Dict[str, Any]:
        counts_arr = np.array(self._det_counts) if self._det_counts else np.zeros(1)
        dpf = {
            "avg": round(float(counts_arr.mean()), 2),
            "min": int(counts_arr.min()),
            "max": int(counts_arr.max()),
        }

        class_dist: Dict[str, int] = {}
        for cls, cnt in sorted(self._class_counts.items()):
            class_dist[self._cls_name(cls)] = cnt

        conf_arr = np.array(list(self._conf_values)) if self._conf_values else np.zeros(1)
        hist_counts, bin_edges = np.histogram(
            conf_arr, bins=self.conf_bins, range=(0.0, 1.0)
        )
        conf_hist = {
            "bins": [round(float(b), 3) for b in bin_edges],
            "counts": [int(c) for c in hist_counts],
        }

        win_class: Dict[str, int] = defaultdict(int)
        for frame_counts in self._window_class_counts:
            for cls, cnt in frame_counts.items():
                win_class[self._cls_name(cls)] += cnt

        result: Dict[str, Any] = {
            "frame_idx": self._frame_idx,
            "total_detections": self._total_detections,
            "detections_per_frame": dpf,
            "class_distribution": class_dist,
            "window_class_distribution": dict(win_class),
            "confidence_histogram": conf_hist,
        }

        if self.tracker is not None:
            result["unique_tracks"] = len(self._unique_track_ids)
            active_arr = np.array(list(self._active_track_ids)) if self._active_track_ids else np.zeros(1)
            result["active_tracks"] = {
                "avg": round(float(active_arr.mean()), 2),
                "current": int(active_arr[-1]) if len(active_arr) > 0 else 0,
            }

        return result

    def get_summary_report(self) -> str:
        """Return a human-readable summary string."""
        r = self.get_results()
        lines = [
            f"=== FlashDet Analytics Report (frame {r['frame_idx']}) ===",
            f"Total detections:     {r['total_detections']}",
            f"Avg detections/frame: {r['detections_per_frame']['avg']}",
            f"Min detections/frame: {r['detections_per_frame']['min']}",
            f"Max detections/frame: {r['detections_per_frame']['max']}",
            "",
            "Class distribution (cumulative):",
        ]
        for name, cnt in r["class_distribution"].items():
            lines.append(f"  {name}: {cnt}")
        if "unique_tracks" in r:
            lines.append(f"\nUnique tracks: {r['unique_tracks']}")
        return "\n".join(lines)

    def reset(self):
        super().reset()
        self._frame_idx = 0
        self._det_counts.clear()
        self._class_counts.clear()
        self._window_class_counts.clear()
        self._conf_values.clear()
        self._unique_track_ids.clear()
        self._active_track_ids.clear()
        self._total_detections = 0

    def _draw_overlay(self, img: np.ndarray):
        counts_arr = np.array(self._det_counts) if self._det_counts else np.zeros(1)
        avg_det = counts_arr.mean()
        cur_det = int(counts_arr[-1]) if len(counts_arr) > 0 else 0

        panel_w, panel_h = 280, 120
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        y0 = 25
        cv2.putText(img, "FlashDet Analytics", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        y0 += 22
        cv2.putText(img, f"Frame: {self._frame_idx}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y0 += 20
        cv2.putText(img, f"Detections: {cur_det}  (avg {avg_det:.1f})", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y0 += 20
        cv2.putText(img, f"Total: {self._total_detections}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        win_class: Dict[int, int] = defaultdict(int)
        for fc in self._window_class_counts:
            for cls, cnt in fc.items():
                win_class[cls] += cnt
        if win_class:
            y0 += 20
            top = sorted(win_class.items(), key=lambda x: -x[1])[:4]
            max_cnt = max(c for _, c in top) or 1
            for cls, cnt in top:
                bar_w = int(cnt / max_cnt * 120)
                name = self._cls_name(cls)[:8]
                cv2.putText(img, name, (10, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                cv2.rectangle(img, (75, y0 - 8), (75 + bar_w, y0 + 2),
                              (0, 200, 200), -1)
                cv2.putText(img, str(cnt), (80 + bar_w, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                y0 += 14

    def _cls_name(self, cls: int) -> str:
        if self.class_names and cls < len(self.class_names):
            return self.class_names[cls]
        return str(cls)
