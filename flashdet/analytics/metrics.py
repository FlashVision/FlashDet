"""Evaluation metrics — mAP, AP per class, F1, precision, recall at various IoU thresholds."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class DetectionMetrics:
    """Compute comprehensive detection evaluation metrics.

    Follows COCO-style evaluation: AP@[0.50:0.95], AP@0.50, AP@0.75,
    plus per-class AP, F1 scores, precision/recall at operating points,
    and size-stratified metrics (small/medium/large).

    Parameters
    ----------
    num_classes : int
        Number of object classes.
    iou_thresholds : sequence[float] | None
        IoU thresholds for AP computation.
        Default: COCO-style [0.50, 0.55, ..., 0.95].
    class_names : list[str] | None
        Human-readable class names.
    """

    AREA_RANGES = {
        "small": (0, 32**2),
        "medium": (32**2, 96**2),
        "large": (96**2, float("inf")),
        "all": (0, float("inf")),
    }

    def __init__(
        self,
        num_classes: int = 80,
        iou_thresholds: Optional[Sequence[float]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.iou_thresholds = np.array(
            iou_thresholds if iou_thresholds is not None
            else [0.50 + 0.05 * i for i in range(10)]
        )
        self.class_names = class_names

        self._predictions: List[Dict[str, Any]] = []
        self._ground_truths: List[Dict[str, Any]] = []

    def add_predictions(
        self,
        image_id: int,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ):
        """Add predictions for one image.

        Parameters
        ----------
        image_id : int
            Unique image identifier.
        boxes : np.ndarray, shape (N, 4)
            Predicted bounding boxes in [x1, y1, x2, y2] format.
        scores : np.ndarray, shape (N,)
            Confidence scores.
        class_ids : np.ndarray, shape (N,)
            Predicted class IDs.
        """
        for i in range(len(boxes)):
            self._predictions.append({
                "image_id": image_id,
                "bbox": boxes[i].tolist(),
                "score": float(scores[i]),
                "class_id": int(class_ids[i]),
            })

    def add_ground_truths(
        self,
        image_id: int,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        is_crowd: Optional[np.ndarray] = None,
    ):
        """Add ground-truth annotations for one image.

        Parameters
        ----------
        image_id : int
            Unique image identifier.
        boxes : np.ndarray, shape (M, 4)
            Ground-truth boxes in [x1, y1, x2, y2] format.
        class_ids : np.ndarray, shape (M,)
            True class IDs.
        is_crowd : np.ndarray | None
            COCO crowd flag per annotation.
        """
        if is_crowd is None:
            is_crowd = np.zeros(len(boxes), dtype=bool)
        for i in range(len(boxes)):
            self._ground_truths.append({
                "image_id": image_id,
                "bbox": boxes[i].tolist(),
                "class_id": int(class_ids[i]),
                "is_crowd": bool(is_crowd[i]),
            })

    def compute(self) -> Dict[str, Any]:
        """Compute all metrics.

        Returns
        -------
        dict
            Comprehensive metrics including mAP, per-class AP, F1, etc.
        """
        results: Dict[str, Any] = {}

        ap_per_class = self._compute_ap_per_class()

        all_aps = [v for v in ap_per_class.values() if v is not None]
        map_all = float(np.mean(all_aps)) if all_aps else 0.0

        ap50_per_class = self._compute_ap_per_class_at_iou(0.50)
        ap75_per_class = self._compute_ap_per_class_at_iou(0.75)

        all_ap50 = [v for v in ap50_per_class.values() if v is not None]
        all_ap75 = [v for v in ap75_per_class.values() if v is not None]

        results["mAP"] = round(map_all, 4)
        results["mAP_50"] = round(float(np.mean(all_ap50)) if all_ap50 else 0.0, 4)
        results["mAP_75"] = round(float(np.mean(all_ap75)) if all_ap75 else 0.0, 4)

        size_maps = {}
        for size_name, (area_min, area_max) in self.AREA_RANGES.items():
            if size_name == "all":
                continue
            ap_s = self._compute_ap_per_class(area_range=(area_min, area_max))
            vals = [v for v in ap_s.values() if v is not None]
            size_maps[f"mAP_{size_name}"] = round(float(np.mean(vals)) if vals else 0.0, 4)
        results.update(size_maps)

        per_class_results = []
        for cls_id in range(self.num_classes):
            name = self._cls_name(cls_id)
            ap = ap_per_class.get(cls_id)
            ap50 = ap50_per_class.get(cls_id)
            if ap is None:
                continue
            prec, rec, f1 = self._precision_recall_f1_at_class(cls_id)
            per_class_results.append({
                "class_id": cls_id,
                "class_name": name,
                "AP": round(ap, 4),
                "AP_50": round(ap50, 4) if ap50 is not None else None,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "num_gt": sum(1 for g in self._ground_truths if g["class_id"] == cls_id),
                "num_pred": sum(1 for p in self._predictions if p["class_id"] == cls_id),
            })

        results["per_class"] = per_class_results
        results["num_predictions"] = len(self._predictions)
        results["num_ground_truths"] = len(self._ground_truths)

        return results

    def summary(self) -> str:
        """Return a human-readable metrics summary."""
        r = self.compute()
        lines = [
            "=" * 70,
            "  FlashDet Detection Evaluation Metrics",
            "=" * 70,
            f"  mAP@[.50:.95]:  {r['mAP']:.4f}",
            f"  mAP@.50:        {r['mAP_50']:.4f}",
            f"  mAP@.75:        {r['mAP_75']:.4f}",
            f"  mAP (small):    {r.get('mAP_small', 0):.4f}",
            f"  mAP (medium):   {r.get('mAP_medium', 0):.4f}",
            f"  mAP (large):    {r.get('mAP_large', 0):.4f}",
            "",
            f"  {'Class':<20} {'AP':>8} {'AP50':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'#GT':>6} {'#Pred':>6}",
            "  " + "-" * 68,
        ]
        for cls in r["per_class"]:
            lines.append(
                f"  {cls['class_name']:<20} {cls['AP']:>8.4f} "
                f"{cls['AP_50'] or 0:>8.4f} {cls['precision']:>8.4f} "
                f"{cls['recall']:>8.4f} {cls['f1']:>8.4f} "
                f"{cls['num_gt']:>6} {cls['num_pred']:>6}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)

    def reset(self):
        """Clear all accumulated predictions and ground truths."""
        self._predictions.clear()
        self._ground_truths.clear()

    # ------------------------------------------------------------------
    # Internal AP computation
    # ------------------------------------------------------------------

    def _compute_ap_per_class(
        self, area_range: Optional[Tuple[float, float]] = None
    ) -> Dict[int, Optional[float]]:
        """Compute mean AP over all IoU thresholds, per class."""
        result = {}
        for cls_id in range(self.num_classes):
            aps = []
            for iou_thresh in self.iou_thresholds:
                ap = self._compute_ap_single(cls_id, iou_thresh, area_range)
                if ap is not None:
                    aps.append(ap)
            result[cls_id] = float(np.mean(aps)) if aps else None
        return result

    def _compute_ap_per_class_at_iou(self, iou_thresh: float) -> Dict[int, Optional[float]]:
        result = {}
        for cls_id in range(self.num_classes):
            result[cls_id] = self._compute_ap_single(cls_id, iou_thresh)
        return result

    def _compute_ap_single(
        self,
        cls_id: int,
        iou_thresh: float,
        area_range: Optional[Tuple[float, float]] = None,
    ) -> Optional[float]:
        """Compute AP for one class at one IoU threshold."""
        gt_by_image = defaultdict(list)
        for g in self._ground_truths:
            if g["class_id"] != cls_id:
                continue
            if area_range:
                area = self._box_area(g["bbox"])
                if area < area_range[0] or area >= area_range[1]:
                    continue
            gt_by_image[g["image_id"]].append(g)

        if not gt_by_image:
            return None

        total_gt = sum(len(v) for v in gt_by_image.values())

        preds = [p for p in self._predictions if p["class_id"] == cls_id]
        preds = sorted(preds, key=lambda x: -x["score"])

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = {img_id: set() for img_id in gt_by_image}

        for i, pred in enumerate(preds):
            img_id = pred["image_id"]
            gts = gt_by_image.get(img_id, [])
            if not gts:
                fp[i] = 1
                continue

            best_iou = 0.0
            best_idx = -1
            for j, gt in enumerate(gts):
                iou = self._compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_thresh and best_idx not in matched[img_id]:
                tp[i] = 1
                matched[img_id].add(best_idx)
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        return self._voc_ap(recalls, precisions)

    def _precision_recall_f1_at_class(self, cls_id: int, iou_thresh: float = 0.5) -> Tuple[float, float, float]:
        """Compute precision, recall, F1 for a single class at given IoU."""
        gt_count = sum(1 for g in self._ground_truths if g["class_id"] == cls_id)
        if gt_count == 0:
            return 0.0, 0.0, 0.0

        preds = [p for p in self._predictions if p["class_id"] == cls_id]
        if not preds:
            return 0.0, 0.0, 0.0

        preds = sorted(preds, key=lambda x: -x["score"])

        gt_by_image = defaultdict(list)
        for g in self._ground_truths:
            if g["class_id"] == cls_id:
                gt_by_image[g["image_id"]].append(g)

        tp = 0
        matched = {img_id: set() for img_id in gt_by_image}
        for pred in preds:
            img_id = pred["image_id"]
            gts = gt_by_image.get(img_id, [])
            best_iou, best_idx = 0.0, -1
            for j, gt in enumerate(gts):
                iou = self._compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou >= iou_thresh and best_idx not in matched.get(img_id, set()):
                tp += 1
                matched.setdefault(img_id, set()).add(best_idx)

        precision = tp / len(preds) if preds else 0.0
        recall = tp / gt_count
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        return precision, recall, f1

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / (union + 1e-12)

    @staticmethod
    def _box_area(box: List[float]) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def _voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute VOC-style AP using all-point interpolation."""
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))
        return ap

    def _cls_name(self, cls_id: int) -> str:
        if self.class_names and cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return f"class_{cls_id}"
