"""Dataset statistics — analyze class distributions, image sizes, bbox geometry."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class DatasetAnalyzer:
    """Compute comprehensive statistics for a detection dataset.

    Supports COCO-format JSON annotations and YOLO-format label directories.

    Parameters
    ----------
    annotation_path : str | Path | None
        Path to a COCO-format JSON annotation file.
    label_dir : str | Path | None
        Path to a directory of YOLO-format ``.txt`` label files.
    image_dir : str | Path | None
        If provided, image dimensions are read directly (slow but accurate).
    class_names : list[str] | None
        Human-readable class names indexed by class ID.
    """

    def __init__(
        self,
        annotation_path: Optional[Union[str, Path]] = None,
        label_dir: Optional[Union[str, Path]] = None,
        image_dir: Optional[Union[str, Path]] = None,
        class_names: Optional[List[str]] = None,
    ):
        if annotation_path is None and label_dir is None:
            raise ValueError("Provide either annotation_path (COCO JSON) or label_dir (YOLO txt)")

        self.annotation_path = Path(annotation_path) if annotation_path else None
        self.label_dir = Path(label_dir) if label_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.class_names = class_names

        self._parsed = False
        self._images: List[Dict[str, Any]] = []
        self._annotations: List[Dict[str, Any]] = []
        self._class_counts: Counter = Counter()
        self._bbox_widths: List[float] = []
        self._bbox_heights: List[float] = []
        self._bbox_areas: List[float] = []
        self._bbox_aspect_ratios: List[float] = []
        self._image_widths: List[int] = []
        self._image_heights: List[int] = []
        self._objects_per_image: List[int] = []

    def analyze(self) -> Dict[str, Any]:
        """Parse annotations and return a full statistics dict."""
        self._parse()
        return {
            "num_images": len(self._images),
            "num_annotations": len(self._annotations),
            "num_classes": len(self._class_counts),
            "class_distribution": self._class_distribution(),
            "image_sizes": self._image_size_stats(),
            "bbox_sizes": self._bbox_size_stats(),
            "objects_per_image": self._objects_per_image_stats(),
            "class_balance": self._class_balance_stats(),
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        stats = self.analyze()
        lines = [
            "=" * 60,
            "  FlashDet Dataset Analysis Report",
            "=" * 60,
            f"  Images:         {stats['num_images']:,}",
            f"  Annotations:    {stats['num_annotations']:,}",
            f"  Classes:        {stats['num_classes']}",
            "",
            "  Objects per image:",
            f"    Mean:  {stats['objects_per_image']['mean']:.1f}",
            f"    Min:   {stats['objects_per_image']['min']}",
            f"    Max:   {stats['objects_per_image']['max']}",
            f"    Std:   {stats['objects_per_image']['std']:.2f}",
            "",
            "  Bounding box sizes (pixels):",
            f"    Width  — mean: {stats['bbox_sizes']['width_mean']:.1f}, "
            f"std: {stats['bbox_sizes']['width_std']:.1f}",
            f"    Height — mean: {stats['bbox_sizes']['height_mean']:.1f}, "
            f"std: {stats['bbox_sizes']['height_std']:.1f}",
            f"    Area   — mean: {stats['bbox_sizes']['area_mean']:.0f}, "
            f"median: {stats['bbox_sizes']['area_median']:.0f}",
            f"    Aspect ratio — mean: {stats['bbox_sizes']['aspect_ratio_mean']:.2f}",
            "",
            "  Class balance:",
            f"    Imbalance ratio (max/min): {stats['class_balance']['imbalance_ratio']:.1f}",
            "",
            "  Top-10 classes:",
        ]
        for name, count in stats["class_distribution"][:10]:
            lines.append(f"    {name:20s}  {count:>7,}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics extractors
    # ------------------------------------------------------------------

    def _class_distribution(self) -> List[Tuple[str, int]]:
        result = []
        for cls_id, count in self._class_counts.most_common():
            name = self._cls_name(cls_id)
            result.append((name, count))
        return result

    def _image_size_stats(self) -> Dict[str, Any]:
        if not self._image_widths:
            return {"width_mean": 0, "height_mean": 0, "width_std": 0, "height_std": 0}
        w = np.array(self._image_widths)
        h = np.array(self._image_heights)
        return {
            "width_mean": float(w.mean()),
            "width_std": float(w.std()),
            "width_min": int(w.min()),
            "width_max": int(w.max()),
            "height_mean": float(h.mean()),
            "height_std": float(h.std()),
            "height_min": int(h.min()),
            "height_max": int(h.max()),
        }

    def _bbox_size_stats(self) -> Dict[str, Any]:
        if not self._bbox_widths:
            return {
                "width_mean": 0, "width_std": 0,
                "height_mean": 0, "height_std": 0,
                "area_mean": 0, "area_median": 0,
                "aspect_ratio_mean": 0,
            }
        bw = np.array(self._bbox_widths)
        bh = np.array(self._bbox_heights)
        ba = np.array(self._bbox_areas)
        ar = np.array(self._bbox_aspect_ratios)
        return {
            "width_mean": float(bw.mean()),
            "width_std": float(bw.std()),
            "height_mean": float(bh.mean()),
            "height_std": float(bh.std()),
            "area_mean": float(ba.mean()),
            "area_median": float(np.median(ba)),
            "area_small_pct": float((ba < 32**2).mean() * 100),
            "area_medium_pct": float(((ba >= 32**2) & (ba < 96**2)).mean() * 100),
            "area_large_pct": float((ba >= 96**2).mean() * 100),
            "aspect_ratio_mean": float(ar.mean()),
            "aspect_ratio_std": float(ar.std()),
        }

    def _objects_per_image_stats(self) -> Dict[str, Any]:
        if not self._objects_per_image:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
        arr = np.array(self._objects_per_image)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "median": float(np.median(arr)),
        }

    def _class_balance_stats(self) -> Dict[str, Any]:
        if not self._class_counts:
            return {"imbalance_ratio": 1.0, "entropy": 0.0}
        counts = np.array(list(self._class_counts.values()), dtype=float)
        imbalance = float(counts.max() / max(counts.min(), 1))
        probs = counts / counts.sum()
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = float(np.log(len(counts)))
        return {
            "imbalance_ratio": round(imbalance, 2),
            "entropy": round(entropy, 4),
            "max_entropy": round(max_entropy, 4),
            "balance_score": round(entropy / max(max_entropy, 1e-12), 4),
        }

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    def _parse(self):
        if self._parsed:
            return
        if self.annotation_path:
            self._parse_coco()
        elif self.label_dir:
            self._parse_yolo()
        self._parsed = True

    def _parse_coco(self):
        with open(self.annotation_path, "r") as f:
            data = json.load(f)

        self._images = data.get("images", [])
        self._annotations = data.get("annotations", [])

        img_lookup = {img["id"]: img for img in self._images}
        objects_per_img: Dict[int, int] = defaultdict(int)

        for img in self._images:
            self._image_widths.append(img.get("width", 0))
            self._image_heights.append(img.get("height", 0))

        for ann in self._annotations:
            cls_id = ann.get("category_id", 0)
            self._class_counts[cls_id] += 1
            objects_per_img[ann["image_id"]] += 1

            bbox = ann.get("bbox", [0, 0, 0, 0])  # COCO: [x, y, w, h]
            w, h = bbox[2], bbox[3]
            self._bbox_widths.append(w)
            self._bbox_heights.append(h)
            self._bbox_areas.append(w * h)
            self._bbox_aspect_ratios.append(w / max(h, 1e-6))

        for img in self._images:
            self._objects_per_image.append(objects_per_img.get(img["id"], 0))

    def _parse_yolo(self):
        label_files = sorted(self.label_dir.glob("*.txt"))
        img_w, img_h = 640, 640  # default if no image_dir

        for lf in label_files:
            if self.image_dir:
                stem = lf.stem
                img_path = self._find_image(stem)
                if img_path:
                    iw, ih = self._read_image_size(img_path)
                    img_w, img_h = iw, ih
                    self._image_widths.append(iw)
                    self._image_heights.append(ih)

            lines = lf.read_text().strip().split("\n")
            lines = [l for l in lines if l.strip()]
            self._objects_per_image.append(len(lines))
            self._images.append({"file_name": lf.stem})

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                abs_w = bw * img_w
                abs_h = bh * img_h
                self._class_counts[cls_id] += 1
                self._bbox_widths.append(abs_w)
                self._bbox_heights.append(abs_h)
                self._bbox_areas.append(abs_w * abs_h)
                self._bbox_aspect_ratios.append(abs_w / max(abs_h, 1e-6))
                self._annotations.append({"class_id": cls_id})

    def _find_image(self, stem: str) -> Optional[Path]:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            p = self.image_dir / (stem + ext)
            if p.exists():
                return p
        return None

    @staticmethod
    def _read_image_size(path: Path) -> Tuple[int, int]:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return 640, 640
        return img.shape[1], img.shape[0]

    def _cls_name(self, cls_id: int) -> str:
        if self.class_names and cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return f"class_{cls_id}"
