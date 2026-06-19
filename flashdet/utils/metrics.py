"""
Evaluation metrics for object detection.
"""

import numpy as np
from typing import List, Dict


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    
    # Union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using the PASCAL VOC all-points interpolation
    (area under the precision-recall curve with a monotone envelope).

    This is NOT 11-point interpolation; it integrates over every recall change
    point, which is more accurate than the older 11-point method.

    Args:
        recalls: Recall values (unsorted subset; sentinel 0 and 1 are added).
        precisions: Corresponding precision values.

    Returns:
        AP value in [0, 1].
    """
    # Add sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Find recall change points
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Sum (recall_change * precision)
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 10
) -> Dict[str, float]:
    """
    Compute mean Average Precision.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
        
    Returns:
        Dictionary with AP per class and mAP
    """
    aps = {}
    
    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        all_preds = []
        all_gts = []
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Get predictions for this class
            pred_mask = pred["labels"] == class_id
            if pred_mask.any():
                for box, score in zip(pred["boxes"][pred_mask], pred["scores"][pred_mask]):
                    all_preds.append({
                        "img_idx": img_idx,
                        "box": box,
                        "score": score,
                        "matched": False
                    })
            
            # Get ground truths for this class
            gt_mask = gt["labels"] == class_id
            if gt_mask.any():
                for box in gt["boxes"][gt_mask]:
                    all_gts.append({
                        "img_idx": img_idx,
                        "box": box,
                        "matched": False
                    })
        
        if len(all_gts) == 0:
            continue
        
        # Sort predictions by score
        all_preds.sort(key=lambda x: x["score"], reverse=True)
        
        # Match predictions to ground truths
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        
        for pred_idx, pred in enumerate(all_preds):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(all_gts):
                if gt["img_idx"] != pred["img_idx"] or gt["matched"]:
                    continue
                
                iou = compute_iou(pred["box"], gt["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp[pred_idx] = 1
                all_gts[best_gt_idx]["matched"] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        recalls = cum_tp / len(all_gts)
        precisions = cum_tp / (cum_tp + cum_fp)
        
        # Compute AP
        aps[class_id] = compute_ap(recalls, precisions)
    
    # Compute mAP
    if len(aps) > 0:
        map_value = np.mean(list(aps.values()))
    else:
        map_value = 0.0
    
    return {
        "mAP": map_value,
        "AP_per_class": aps
    }


def compute_map_coco_summary(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
) -> Dict[str, float]:
    """
    COCO-style detection summary: mAP@0.5, mAP@0.5:0.95 (mean over IoU 0.5..0.95 step 0.05),
    and AP_per_class at IoU 0.5 only.
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    map_per_iou = []
    ap_per_class_at_50 = {}
    for thr in iou_thresholds:
        r = compute_map(
            predictions,
            ground_truths,
            iou_threshold=float(thr),
            num_classes=num_classes,
        )
        map_per_iou.append(r["mAP"])
        if abs(float(thr) - 0.5) < 1e-9:
            ap_per_class_at_50 = r["AP_per_class"]

    return {
        "mAP@0.5": map_per_iou[0],
        "mAP@0.5:0.95": float(np.mean(map_per_iou)),
        "AP_per_class": ap_per_class_at_50,
    }


class MeterBuffer:
    """Buffer for computing running averages."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.values = {}
    
    def update(self, **kwargs):
        """Update buffer with new values."""
        for key, value in kwargs.items():
            if key not in self.values:
                self.values[key] = []
            self.values[key].append(value)
            if len(self.values[key]) > self.window_size:
                self.values[key].pop(0)
    
    def get_avg(self, key: str) -> float:
        """Get average value for key."""
        if key not in self.values or len(self.values[key]) == 0:
            return 0.0
        return sum(self.values[key]) / len(self.values[key])
    
    def get_latest(self, key: str) -> float:
        """Get latest value for key."""
        if key not in self.values or len(self.values[key]) == 0:
            return 0.0
        return self.values[key][-1]
    
    def reset(self):
        """Reset all values."""
        self.values = {}
