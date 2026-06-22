"""
End-to-End Detection Loss with Progressive Loss (ProgLoss) for YOLO26-based FlashDet.

DFL-free: box regression uses direct 4-value LTRB prediction (no distribution).
Loss components per head:
    - Classification: BCE with soft-label targets from TAL alignment
    - Box regression: CIoU loss on decoded xyxy boxes
    - L1 distance: normalized L1 on LTRB distances (replaces DFL)

ProgLoss linearly shifts emphasis from the one-to-many head to the
one-to-one head over training:
    L_total = alpha(t) * L_o2m + (1 - alpha(t)) * L_o2o

Reference:
    Ultralytics YOLO26 (2026), Sections 3.2.2, 3.3.2.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from flashdet.models.assignment.stal import STALAssigner, _bbox_iou_aligned


def _make_anchor_grid(
    feat_sizes: List[Tuple[int, int]],
    strides: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build anchor center grid and stride tensor for all FPN levels.

    Returns:
        anchor_centers: [total_anchors, 2] (x, y) in pixel space.
        anchor_strides: [total_anchors, 1] stride per anchor.
    """
    centers_list = []
    strides_list = []
    for (h, w), stride in zip(feat_sizes, strides):
        shift_x = (torch.arange(w, device=device) + 0.5) * stride
        shift_y = (torch.arange(h, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
        centers_list.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1))
        strides_list.append(torch.full((h * w, 1), stride, device=device, dtype=torch.float32))
    return torch.cat(centers_list, dim=0), torch.cat(strides_list, dim=0)


def _decode_ltrb(
    anchor_centers: torch.Tensor,
    anchor_strides: torch.Tensor,
    reg_pred: torch.Tensor,
) -> torch.Tensor:
    """Decode LTRB distances to xyxy boxes.

    Args:
        anchor_centers: [N, 2]
        anchor_strides: [N, 1]
        reg_pred: [N, 4] raw regression output (will be exponentiated * stride).

    Returns:
        decoded_bboxes: [N, 4] in xyxy format.
    """
    ltrb = F.softplus(reg_pred, beta=1.0) * anchor_strides  # [N, 4]
    x1 = anchor_centers[:, 0] - ltrb[:, 0]
    y1 = anchor_centers[:, 1] - ltrb[:, 1]
    x2 = anchor_centers[:, 0] + ltrb[:, 2]
    y2 = anchor_centers[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _compute_branch_loss(
    cls_preds: torch.Tensor,
    reg_preds: torch.Tensor,
    anchor_centers: torch.Tensor,
    anchor_strides: torch.Tensor,
    gt_bboxes_list: List[torch.Tensor],
    gt_labels_list: List[torch.Tensor],
    num_classes: int,
    assigner: STALAssigner,
    box_weight: float = 7.5,
    cls_weight: float = 0.5,
    l1_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute detection loss for a single branch (o2o or o2m).

    Args:
        cls_preds: [B, N, num_classes] sigmoid classification logits.
        reg_preds: [B, N, 4] raw LTRB regression outputs.
        anchor_centers: [N, 2]
        anchor_strides: [N, 1]
        gt_bboxes_list: List of [M_i, 4] per image.
        gt_labels_list: List of [M_i] per image.
        num_classes: Number of classes.
        assigner: STALAssigner instance.
        box_weight: CIoU box loss weight.
        cls_weight: Classification BCE loss weight.
        l1_weight: L1 distance loss weight.

    Returns:
        total_loss, loss_states dict.
    """
    device = cls_preds.device
    B = cls_preds.shape[0]
    total_cls_loss = torch.tensor(0.0, device=device)
    total_box_loss = torch.tensor(0.0, device=device)
    total_l1_loss = torch.tensor(0.0, device=device)
    total_pos = 0

    for b in range(B):
        cls_pred_b = cls_preds[b]        # [N, num_classes]
        reg_pred_b = reg_preds[b]        # [N, 4]

        decoded_bboxes = _decode_ltrb(anchor_centers, anchor_strides, reg_pred_b)
        cls_scores = cls_pred_b.sigmoid()

        gt_bboxes = gt_bboxes_list[b]
        gt_labels = gt_labels_list[b]

        if gt_bboxes.shape[0] == 0:
            # No GT: all-negative BCE
            target = torch.zeros_like(cls_pred_b)
            total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
                cls_pred_b, target, reduction="sum"
            )
            continue

        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = assigner.assign(
            anchor_centers, cls_scores, decoded_bboxes, gt_bboxes, gt_labels,
        )

        n_pos = fg_mask.sum().item()
        total_pos += n_pos

        # Classification loss (BCE with soft-label targets)
        total_cls_loss = total_cls_loss + F.binary_cross_entropy_with_logits(
            cls_pred_b, assigned_scores, reduction="sum"
        )

        if n_pos > 0:
            pos_decoded = decoded_bboxes[fg_mask]   # [n_pos, 4]
            pos_target = assigned_bboxes[fg_mask]   # [n_pos, 4]
            pos_reg = reg_pred_b[fg_mask]           # [n_pos, 4]

            # CIoU box loss (CIoU range is [-1, 1], loss = 1 - ciou)
            ciou = _bbox_iou_aligned(pos_decoded, pos_target)
            box_loss = (1 - ciou).clamp(min=0).sum()
            total_box_loss = total_box_loss + box_loss

            # L1 distance loss on normalized LTRB
            pos_centers = anchor_centers[fg_mask]    # [n_pos, 2]
            pos_strides = anchor_strides[fg_mask]    # [n_pos, 1]
            target_l = (pos_centers[:, 0] - pos_target[:, 0]) / pos_strides[:, 0]
            target_t = (pos_centers[:, 1] - pos_target[:, 1]) / pos_strides[:, 0]
            target_r = (pos_target[:, 2] - pos_centers[:, 0]) / pos_strides[:, 0]
            target_b = (pos_target[:, 3] - pos_centers[:, 1]) / pos_strides[:, 0]
            target_ltrb = torch.stack([target_l, target_t, target_r, target_b], dim=-1)
            pred_ltrb = F.softplus(pos_reg, beta=1.0)
            l1_loss = F.l1_loss(pred_ltrb, target_ltrb.clamp(min=0), reduction="sum")
            total_l1_loss = total_l1_loss + l1_loss

    n_pos = max(total_pos, 1)
    n_total = B * cls_preds.shape[1]
    loss_cls = cls_weight * total_cls_loss / max(n_total, 1)
    loss_box = box_weight * total_box_loss / n_pos
    loss_l1 = l1_weight * total_l1_loss / n_pos

    total = loss_cls + loss_box + loss_l1

    return total, {
        "loss_cls": loss_cls.detach(),
        "loss_box": loss_box.detach(),
        "loss_l1": loss_l1.detach(),
        "num_pos": total_pos,
    }


class E2EDetectionLoss:
    """End-to-End detection loss with ProgLoss scheduling.

    Supports dual-head training:
        L_total = alpha(t) * L_o2m + (1 - alpha(t)) * L_o2o

    Args:
        num_classes: Number of object classes.
        strides: Feature pyramid strides.
        alpha_init: Initial one-to-many weight. Default: 1.0.
        alpha_final: Final one-to-many weight. Default: 0.0.
        o2m_topk: Top-k for one-to-many assigner. Default: 10.
        o2o_topk: Top-k for one-to-one assigner (then filtered to 1). Default: 7.
        box_weight: CIoU loss weight. Default: 7.5.
        cls_weight: BCE classification loss weight. Default: 0.5.
        l1_weight: L1 distance loss weight. Default: 1.0.
    """

    def __init__(
        self,
        num_classes: int,
        strides: Tuple[int, ...] = (8, 16, 32),
        alpha_init: float = 1.0,
        alpha_final: float = 0.0,
        o2m_topk: int = 10,
        o2o_topk: int = 7,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        l1_weight: float = 1.0,
    ):
        self.num_classes = num_classes
        self.strides = strides
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight

        self.o2m_assigner = STALAssigner(topk=o2m_topk, strides=strides)
        self.o2o_assigner = STALAssigner(topk=o2o_topk, strides=strides)

    def prog_alpha(self, epoch: int, total_epochs: int) -> float:
        """Compute ProgLoss alpha(t) — linearly decreasing from init to final."""
        if total_epochs <= 1:
            return self.alpha_final
        t = min(epoch, total_epochs - 1)
        ratio = t / max(total_epochs - 1, 1)
        return max((1 - ratio) * (self.alpha_init - self.alpha_final) + self.alpha_final, 0.0)

    def __call__(
        self,
        o2o_cls: torch.Tensor,
        o2o_reg: torch.Tensor,
        o2m_cls: torch.Tensor,
        o2m_reg: torch.Tensor,
        gt_bboxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
        feat_sizes: List[Tuple[int, int]],
        epoch: int = 0,
        total_epochs: int = 100,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the full E2E detection loss with ProgLoss.

        Args:
            o2o_cls: [B, N, num_classes] one-to-one cls logits.
            o2o_reg: [B, N, 4] one-to-one reg predictions.
            o2m_cls: [B, N, num_classes] one-to-many cls logits.
            o2m_reg: [B, N, 4] one-to-many reg predictions.
            gt_bboxes_list: List of [M_i, 4] per image (xyxy, float).
            gt_labels_list: List of [M_i] per image (long).
            feat_sizes: [(H, W)] per FPN level.
            epoch: Current training epoch.
            total_epochs: Total training epochs.

        Returns:
            total_loss, loss_states dict.
        """
        device = o2o_cls.device
        anchor_centers, anchor_strides = _make_anchor_grid(
            feat_sizes, list(self.strides), device
        )

        alpha = self.prog_alpha(epoch, total_epochs)

        # One-to-Many branch
        o2m_loss, o2m_states = _compute_branch_loss(
            o2m_cls, o2m_reg, anchor_centers, anchor_strides,
            gt_bboxes_list, gt_labels_list, self.num_classes,
            self.o2m_assigner, self.box_weight, self.cls_weight, self.l1_weight,
        )

        # One-to-One branch
        o2o_loss, o2o_states = _compute_branch_loss(
            o2o_cls, o2o_reg, anchor_centers, anchor_strides,
            gt_bboxes_list, gt_labels_list, self.num_classes,
            self.o2o_assigner, self.box_weight, self.cls_weight, self.l1_weight,
        )

        # ProgLoss: weighted combination
        total = alpha * o2m_loss + (1 - alpha) * o2o_loss

        states = {
            "loss_total": total.detach(),
            "o2m_loss": o2m_loss.detach(),
            "o2o_loss": o2o_loss.detach(),
            "prog_alpha": alpha,
            "o2m_cls": o2m_states["loss_cls"],
            "o2m_box": o2m_states["loss_box"],
            "o2m_l1": o2m_states["loss_l1"],
            "o2m_pos": o2m_states["num_pos"],
            "o2o_cls": o2o_states["loss_cls"],
            "o2o_box": o2o_states["loss_box"],
            "o2o_l1": o2o_states["loss_l1"],
            "o2o_pos": o2o_states["num_pos"],
        }

        return total, states
