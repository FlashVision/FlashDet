"""
Small-Target-Aware Label Assignment (STAL) for YOLO26-based FlashDet.

Implements Task-Aligned Assignment with small-target protection:
  - TAL scoring: alignment = cls_score^alpha * iou^beta
  - STAL: For tiny ground-truth boxes (w or h < s_min), temporarily
    expands them to s_ref during candidate selection only, ensuring
    at least a few anchor centers fall inside and preventing
    zero-positive supervision.

Reference:
    Ultralytics YOLO26 (2026), Section 3.3.3.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def _bbox_iou_aligned(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """CIoU between aligned pairs [N,4] in xyxy format."""
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1 + area2 - inter + eps
    iou = inter / union

    enclose_x1 = torch.min(box1[:, 0], box2[:, 0])
    enclose_y1 = torch.min(box1[:, 1], box2[:, 1])
    enclose_x2 = torch.max(box1[:, 2], box2[:, 2])
    enclose_y2 = torch.max(box1[:, 3], box2[:, 3])
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
    cx1, cy1 = (box1[:, 0] + box1[:, 2]) / 2, (box1[:, 1] + box1[:, 3]) / 2
    cx2, cy2 = (box2[:, 0] + box2[:, 2]) / 2, (box2[:, 1] + box2[:, 3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    w1, h1 = box1[:, 2] - box1[:, 0], box1[:, 3] - box1[:, 1]
    w2, h2 = box2[:, 2] - box2[:, 0], box2[:, 3] - box2[:, 1]
    import math
    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha_ciou = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha_ciou * v
    return ciou


def _pairwise_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU between [N,4] and [M,4] in xyxy format -> [N,M]."""
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]))[:, None]
    area2 = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1]))[None, :]
    return inter / (area1 + area2 - inter + eps)


class STALAssigner:
    """Task-Aligned Label Assigner with Small-Target-Aware filtering (STAL).

    For each ground-truth box, TAL computes an alignment metric:
        alignment = cls_score^alpha * iou^beta
    and selects the top-k anchors. STAL modifies only the initial
    candidate-selection step: if a GT box is smaller than the
    smallest stride, it is temporarily expanded to s_ref so more
    anchors fall inside and can be scored by TAL.

    When ``one_to_one=True`` (used by the o2o head), a greedy bipartite
    matching step ensures each GT is matched to exactly one anchor and
    each anchor matches at most one GT — the prerequisite for NMS-free
    inference.

    Args:
        topk: Number of candidates per GT for one-to-many assignment.
        alpha: Classification alignment exponent.
        beta: IoU alignment exponent.
        strides: Feature pyramid strides (e.g. [8, 16, 32]).
        s_ref: Reference size for STAL expansion. Defaults to strides[1].
        one_to_one: Enforce 1:1 GT-to-anchor matching (for o2o head).
        eps: Numerical stability constant.
    """

    def __init__(
        self,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
        strides: Tuple[int, ...] = (8, 16, 32),
        s_ref: Optional[int] = None,
        one_to_one: bool = False,
        eps: float = 1e-9,
    ):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.strides = strides
        self.s_min = strides[0]
        self.s_ref = s_ref if s_ref is not None else (strides[1] if len(strides) > 1 else strides[0] * 2)
        self.one_to_one = one_to_one
        self.eps = eps

    @torch.no_grad()
    def assign(
        self,
        anchor_centers: torch.Tensor,
        cls_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign ground truths to anchors using TAL + STAL.

        Args:
            anchor_centers: [N_anchors, 2] anchor center coordinates (x, y) in pixel space.
            cls_scores: [N_anchors, num_classes] predicted classification scores (sigmoid).
            pred_bboxes: [N_anchors, 4] predicted boxes in xyxy pixel coords.
            gt_bboxes: [N_gt, 4] ground truth boxes in xyxy pixel coords.
            gt_labels: [N_gt] ground truth class labels (long).

        Returns:
            assigned_labels: [N_anchors] class index for positives, num_classes for negatives.
            assigned_bboxes: [N_anchors, 4] target boxes for positives (zeros for negatives).
            assigned_scores: [N_anchors, num_classes] soft label targets.
            fg_mask: [N_anchors] boolean mask for positive anchors.
        """
        device = anchor_centers.device
        n_anchors = anchor_centers.shape[0]
        num_classes = cls_scores.shape[1]
        n_gt = gt_bboxes.shape[0]

        assigned_labels = torch.full((n_anchors,), num_classes, dtype=torch.long, device=device)
        assigned_bboxes = torch.zeros((n_anchors, 4), device=device)
        assigned_scores = torch.zeros((n_anchors, num_classes), device=device)
        fg_mask = torch.zeros(n_anchors, dtype=torch.bool, device=device)

        if n_gt == 0:
            return assigned_labels, assigned_bboxes, assigned_scores, fg_mask

        # --- STAL: expand tiny GT boxes for candidate selection only ---
        gt_cxcywh = _xyxy_to_cxcywh(gt_bboxes)
        sel_cxcywh = gt_cxcywh.clone()
        tiny_w = sel_cxcywh[:, 2] < self.s_min
        tiny_h = sel_cxcywh[:, 3] < self.s_min
        sel_cxcywh[:, 2] = torch.where(tiny_w, torch.tensor(float(self.s_ref), device=device), sel_cxcywh[:, 2])
        sel_cxcywh[:, 3] = torch.where(tiny_h, torch.tensor(float(self.s_ref), device=device), sel_cxcywh[:, 3])

        # Convert expanded boxes back to xyxy for candidate selection
        sel_x1 = sel_cxcywh[:, 0] - sel_cxcywh[:, 2] / 2
        sel_y1 = sel_cxcywh[:, 1] - sel_cxcywh[:, 3] / 2
        sel_x2 = sel_cxcywh[:, 0] + sel_cxcywh[:, 2] / 2
        sel_y2 = sel_cxcywh[:, 1] + sel_cxcywh[:, 3] / 2
        sel_bboxes = torch.stack([sel_x1, sel_y1, sel_x2, sel_y2], dim=-1)

        # --- Candidate selection: anchor center inside (STAL-expanded) GT box ---
        # anchor_centers [N,2] vs sel_bboxes [M,4] -> [N,M]
        lt = anchor_centers[:, None, :] - sel_bboxes[None, :, :2]   # [N, M, 2]
        rb = sel_bboxes[None, :, 2:] - anchor_centers[:, None, :]   # [N, M, 2]
        deltas = torch.cat([lt, rb], dim=-1)                         # [N, M, 4]
        is_in_gt = deltas.min(dim=-1).values > 0                     # [N, M]

        # --- Pairwise IoU (using real GT boxes, not expanded) ---
        pair_iou = _pairwise_iou(pred_bboxes, gt_bboxes)  # [N, M]

        # --- TAL alignment metric ---
        # Gather cls scores at GT class positions: [N, M]
        gt_cls_idx = gt_labels[None, :].expand(n_anchors, -1)       # [N, M]
        cls_at_gt = cls_scores.gather(1, gt_cls_idx.clamp(0, num_classes - 1))  # [N, M]

        alignment = cls_at_gt.pow(self.alpha) * pair_iou.pow(self.beta)  # [N, M]

        # Mask out anchors not inside GT
        alignment[~is_in_gt] = 0

        # --- Top-k selection per GT ---
        topk = min(self.topk, n_anchors)
        topk_vals, topk_idxs = alignment.topk(topk, dim=0)  # [topk, M]

        # Build matching matrix
        matching_matrix = torch.zeros_like(alignment)  # [N, M]
        for gt_idx in range(n_gt):
            mask = topk_vals[:, gt_idx] > 0
            if mask.any():
                matching_matrix[topk_idxs[mask, gt_idx], gt_idx] = alignment[topk_idxs[mask, gt_idx], gt_idx]

        # Resolve conflicts: if anchor matched to multiple GTs, keep highest alignment
        multi_match = (matching_matrix > 0).sum(dim=1) > 1
        if multi_match.any():
            max_gt = matching_matrix[multi_match].argmax(dim=1)
            matching_matrix[multi_match] = 0
            matching_matrix[multi_match, max_gt] = 1.0

        # One-to-one: each GT gets exactly one anchor (greedy by alignment)
        if self.one_to_one:
            new_matrix = torch.zeros_like(matching_matrix)
            used_anchors = torch.zeros(n_anchors, dtype=torch.bool, device=device)
            best_vals, _ = matching_matrix.max(dim=0)
            gt_order = best_vals.argsort(descending=True)
            for g in gt_order:
                g = g.item()
                if best_vals[g] <= 0:
                    break
                col = matching_matrix[:, g].clone()
                col[used_anchors] = 0
                a = col.argmax().item()
                if col[a] > 0:
                    new_matrix[a, g] = matching_matrix[a, g]
                    used_anchors[a] = True
            matching_matrix = new_matrix

        # --- Final assignment ---
        fg_mask = (matching_matrix > 0).any(dim=1)  # [N]
        if fg_mask.sum() == 0:
            return assigned_labels, assigned_bboxes, assigned_scores, fg_mask

        matched_gt_idx = matching_matrix[fg_mask].argmax(dim=1)  # [n_pos]
        assigned_labels[fg_mask] = gt_labels[matched_gt_idx]
        assigned_bboxes[fg_mask] = gt_bboxes[matched_gt_idx]

        pos_labels_onehot = F.one_hot(gt_labels[matched_gt_idx], num_classes).float()

        # Soft labels for both heads — matches Ultralytics TaskAlignedAssigner
        # target_scores = one_hot * norm_align_metric
        # where norm_align_metric = (alignment * max_overlap_per_gt) / (max_alignment_per_gt + eps)
        pos_alignment = matching_matrix[fg_mask].max(dim=1).values
        pos_iou = pair_iou[fg_mask, matched_gt_idx]
        max_align_per_gt = matching_matrix.max(dim=0).values
        gt_max = max_align_per_gt[matched_gt_idx].clamp(min=self.eps)
        norm_align = (pos_alignment / gt_max * pos_iou).clamp(min=0.05)
        assigned_scores[fg_mask] = pos_labels_onehot * norm_align[:, None]

        # #region agent log
        import os as _os4, json as _json4, time as _time4
        _lp4 = _os4.path.join(_os4.path.dirname(_os4.path.dirname(_os4.path.dirname(_os4.path.abspath(__file__)))), "debug-387c01.log")
        if not hasattr(self, '_dbg_assign_logged'):
            self._dbg_assign_logged = True
            n_pos = fg_mask.sum().item()
            try:
                _data = {"branch":"o2o" if self.one_to_one else "o2m","n_gt":n_gt,"n_pos":n_pos,"score_sum":round(assigned_scores.sum().item(),4),"norm_align_mean":round(norm_align.mean().item(),4) if n_pos>0 else 0,"iou_mean":round(pos_iou.mean().item(),4) if n_pos>0 else 0}
                if n_pos > 0:
                    _sc = assigned_scores[fg_mask].sum(-1)
                    _data["score_mean"] = round(_sc.mean().item(),4)
                    _data["score_min"] = round(_sc.min().item(),4)
                    _data["score_max"] = round(_sc.max().item(),4)
                with open(_lp4, "a") as _f4:
                    _f4.write(_json4.dumps({"sessionId":"387c01","hypothesisId":"H","location":"stal.py:assign","message":"label_assignment_postfix","data":_data,"timestamp":int(_time4.time()*1000)}) + "\n")
            except: pass
        # #endregion

        return assigned_labels, assigned_bboxes, assigned_scores, fg_mask
