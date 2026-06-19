"""
Dynamic Soft Label Assigner for FlashDet.
Matches official FlashDet implementation exactly.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from ...utils.box_utils import bbox_overlaps


@dataclass
class AssignResult:
    """Assignment result container."""
    num_gts: int
    gt_inds: torch.Tensor
    max_overlaps: torch.Tensor
    labels: Optional[torch.Tensor] = None


class DynamicSoftLabelAssigner:
    """Dynamic Soft Label Assigner (DSLA) for target assignment.

    Uses OTA-style dynamic-k matching with a combined classification + IoU cost.

    Args:
        topk (int): Number of top-IoU candidates used to compute dynamic k.
            Default: 13.
        iou_factor (float): Weight of the IoU cost term. Default: 3.0.
        ignore_iof_thr (float): If > 0, priors whose max IoF with any
            ``gt_bboxes_ignore`` box exceeds this threshold are marked as
            ignored (``gt_inds = -1``). Default: -1 (disabled).
    """

    def __init__(self, topk: int = 13, iou_factor: float = 3.0,
                 ignore_iof_thr: float = -1):
        self.topk = topk
        self.iou_factor = iou_factor
        self.ignore_iof_thr = ignore_iof_thr

    def assign(
        self,
        pred_scores: torch.Tensor,
        priors: torch.Tensor,
        decoded_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes_ignore: torch.Tensor = None,
    ) -> AssignResult:
        """Assign ground truths to priors.

        Args:
            pred_scores: Classification scores [num_priors, num_classes].
            priors: Prior points [num_priors, 4] in [cx, cy, stride_w, stride_h].
            decoded_bboxes: Predicted boxes [num_priors, 4] xyxy.
            gt_bboxes: Ground truth boxes [num_gts, 4] xyxy.
            gt_labels: Ground truth labels [num_gts].
            gt_bboxes_ignore: Ignored gt boxes [num_ignore, 4] (optional).

        Returns:
            AssignResult with gt_inds (0=bg, -1=ignore, >0=fg), max_overlaps,
            and labels.
        """
        INF = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)

        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)

        prior_center = priors[:, :2]
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]
        deltas = torch.cat([lt_, rb_], dim=-1)
        is_in_gts = deltas.min(dim=-1).values > 0
        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        # Empty assignment: no gts, no boxes, or no valid priors
        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full(
                (num_bboxes,), -1, dtype=torch.long
            )
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + 1e-7)

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()

        cls_cost = F.binary_cross_entropy_with_logits(
            valid_pred_scores, soft_label, reduction="none"
        ) * scale_factor.abs().pow(2.0)
        cls_cost = cls_cost.sum(dim=-1)

        cost_matrix = cls_cost + iou_cost * self.iou_factor

        matched_pred_ious, matched_gt_inds = self._dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask
        )

        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()

        max_overlaps = assigned_gt_inds.new_full(
            (num_bboxes,), -INF, dtype=torch.float32
        )
        max_overlaps[valid_mask] = matched_pred_ious

        # Mark priors that highly overlap with ignored regions.
        # Must use full decoded_bboxes (not valid_decoded_bbox) so the mask
        # shape [num_bboxes] matches assigned_gt_inds.
        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and num_bboxes > 0
        ):
            ignore_overlaps = bbox_overlaps(
                decoded_bboxes, gt_bboxes_ignore, mode="iof"
            )
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            assigned_gt_inds[ignore_idxs] = -1

        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, assigned_labels)

    def _dynamic_k_matching(
        self,
        cost: torch.Tensor,
        pairwise_ious: torch.Tensor,
        num_gt: int,
        valid_mask: torch.Tensor,
    ):
        """Dynamic-k matching using sum-of-topk IoU as per-GT candidate count.

        Args:
            cost: Cost matrix [num_valid, num_gt].
            pairwise_ious: IoU matrix [num_valid, num_gt].
            num_gt: Number of ground truth boxes.
            valid_mask: Boolean mask of valid priors [num_priors].

        Returns:
            Tuple of (matched_pred_ious [num_fg], matched_gt_inds [num_fg]).
        """
        matching_matrix = torch.zeros_like(cost)

        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0

        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]

        return matched_pred_ious, matched_gt_inds
