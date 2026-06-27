"""Detection error analysis — categorize and diagnose prediction errors."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class DetectionErrorAnalyzer:
    """Analyze detection errors by category (inspired by TIDE).

    Categorizes every false positive and missed detection into one of:

    - **Classification error**: correct localization, wrong class
    - **Localization error**: correct class, IoU between 0.1 and threshold
    - **Both (cls+loc)**: wrong class AND poor localization
    - **Duplicate**: correct match but GT already matched
    - **Background**: no nearby GT at all (IoU < 0.1)
    - **Missed**: GT not matched by any prediction

    Parameters
    ----------
    num_classes : int
        Number of classes.
    iou_threshold : float
        IoU threshold for a correct match (default: 0.5).
    score_threshold : float
        Minimum confidence to consider a prediction (default: 0.0).
    class_names : list[str] | None
        Human-readable class names.
    """

    def __init__(
        self,
        num_classes: int = 80,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0,
        class_names: Optional[List[str]] = None,
    ):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
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
        """Add predictions for one image."""
        for i in range(len(boxes)):
            if float(scores[i]) < self.score_threshold:
                continue
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
    ):
        """Add ground-truth annotations for one image."""
        for i in range(len(boxes)):
            self._ground_truths.append({
                "image_id": image_id,
                "bbox": boxes[i].tolist(),
                "class_id": int(class_ids[i]),
            })

    def analyze(self) -> Dict[str, Any]:
        """Run the error analysis.

        Returns
        -------
        dict
            Keys:
            - ``"errors"``: list of error dicts with type, pred, gt info
            - ``"summary"``: counts per error category
            - ``"per_class"``: per-class error breakdown
            - ``"confusion_pairs"``: top class confusion pairs
        """
        errors: List[Dict[str, Any]] = []
        matched_gt: Dict[int, Dict[int, bool]] = defaultdict(lambda: defaultdict(bool))

        gt_by_image = defaultdict(list)
        for i, g in enumerate(self._ground_truths):
            gt_by_image[g["image_id"]].append((i, g))

        preds_sorted = sorted(self._predictions, key=lambda x: -x["score"])

        for pred in preds_sorted:
            img_id = pred["image_id"]
            gts = gt_by_image.get(img_id, [])

            if not gts:
                errors.append(self._make_error("background", pred))
                continue

            best_iou = 0.0
            best_gt_idx = -1
            best_gt = None
            for gt_idx, gt in gts:
                iou = self._iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt = gt

            if best_iou >= self.iou_threshold:
                if matched_gt[img_id][best_gt_idx]:
                    errors.append(self._make_error("duplicate", pred, best_gt, best_iou))
                elif pred["class_id"] == best_gt["class_id"]:
                    matched_gt[img_id][best_gt_idx] = True
                    # True positive — not an error
                else:
                    matched_gt[img_id][best_gt_idx] = True
                    errors.append(self._make_error("classification", pred, best_gt, best_iou))
            elif best_iou >= 0.1:
                if pred["class_id"] == best_gt["class_id"]:
                    errors.append(self._make_error("localization", pred, best_gt, best_iou))
                else:
                    errors.append(self._make_error("cls_and_loc", pred, best_gt, best_iou))
            else:
                errors.append(self._make_error("background", pred, best_gt, best_iou))

        for img_id, gt_list in gt_by_image.items():
            for gt_idx, gt in gt_list:
                if not matched_gt[img_id][gt_idx]:
                    errors.append(self._make_error("missed", None, gt))

        summary = defaultdict(int)
        per_class_errors = defaultdict(lambda: defaultdict(int))
        confusion_pairs = defaultdict(int)

        for err in errors:
            summary[err["type"]] += 1
            cls = err.get("gt_class") or err.get("pred_class")
            if cls is not None:
                per_class_errors[cls][err["type"]] += 1
            if err["type"] == "classification":
                pair = (err["pred_class"], err["gt_class"])
                confusion_pairs[pair] += 1

        top_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:20]
        top_confusions_readable = [
            {
                "predicted": self._cls_name(p),
                "actual": self._cls_name(a),
                "count": c,
            }
            for (p, a), c in top_confusions
        ]

        per_class_result = {}
        for cls_id, err_counts in per_class_errors.items():
            per_class_result[self._cls_name(cls_id)] = dict(err_counts)

        return {
            "errors": errors,
            "summary": dict(summary),
            "per_class": per_class_result,
            "confusion_pairs": top_confusions_readable,
            "total_predictions": len(self._predictions),
            "total_ground_truths": len(self._ground_truths),
            "total_errors": len(errors),
        }

    def summary(self) -> str:
        """Return a human-readable error analysis summary."""
        result = self.analyze()
        s = result["summary"]
        total = result["total_errors"]

        lines = [
            "=" * 60,
            "  FlashDet Detection Error Analysis",
            "=" * 60,
            f"  Total predictions:   {result['total_predictions']}",
            f"  Total ground truths: {result['total_ground_truths']}",
            f"  Total errors:        {total}",
            "",
            "  Error breakdown:",
            f"    Classification:  {s.get('classification', 0):>5} "
            f"({s.get('classification', 0) / max(total, 1) * 100:.1f}%)",
            f"    Localization:    {s.get('localization', 0):>5} "
            f"({s.get('localization', 0) / max(total, 1) * 100:.1f}%)",
            f"    Cls + Loc:       {s.get('cls_and_loc', 0):>5} "
            f"({s.get('cls_and_loc', 0) / max(total, 1) * 100:.1f}%)",
            f"    Duplicate:       {s.get('duplicate', 0):>5} "
            f"({s.get('duplicate', 0) / max(total, 1) * 100:.1f}%)",
            f"    Background:      {s.get('background', 0):>5} "
            f"({s.get('background', 0) / max(total, 1) * 100:.1f}%)",
            f"    Missed:          {s.get('missed', 0):>5} "
            f"({s.get('missed', 0) / max(total, 1) * 100:.1f}%)",
        ]

        if result["confusion_pairs"]:
            lines.extend(["", "  Top confusion pairs:"])
            for pair in result["confusion_pairs"][:10]:
                lines.append(
                    f"    {pair['predicted']:>15} → {pair['actual']:<15}  ({pair['count']}x)"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self):
        """Clear all stored predictions and ground truths."""
        self._predictions.clear()
        self._ground_truths.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_error(
        error_type: str,
        pred: Optional[Dict] = None,
        gt: Optional[Dict] = None,
        iou: float = 0.0,
    ) -> Dict[str, Any]:
        err: Dict[str, Any] = {"type": error_type, "iou": round(iou, 4)}
        if pred:
            err["pred_bbox"] = pred["bbox"]
            err["pred_class"] = pred["class_id"]
            err["pred_score"] = pred["score"]
            err["image_id"] = pred["image_id"]
        if gt:
            err["gt_bbox"] = gt["bbox"]
            err["gt_class"] = gt["class_id"]
            if "image_id" not in err:
                err["image_id"] = gt["image_id"]
        return err

    @staticmethod
    def _iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / (union + 1e-12)

    def _cls_name(self, cls_id: int) -> str:
        if self.class_names and cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return f"class_{cls_id}"
