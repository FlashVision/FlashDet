"""Shared simple detection loss for YOLO-family models (v9, v10, v11).

Implements a basic anchor-free assignment + classification (BCE) and
regression (DFL + L1) loss that is sufficient for training. Production
YOLO implementations use more sophisticated assignment strategies
(TAL, SimOTA, etc.), but this keeps the codebase self-contained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


def compute_yolo_loss(
    preds: List[torch.Tensor],
    gt_meta: Dict,
    num_classes: int,
    strides: List[int] = None,
    reg_max: int = 16,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a simple detection loss from YOLO-style multi-scale predictions.

    Args:
        preds: List of ``[B, num_classes + 4*(reg_max+1), H, W]`` tensors,
            one per FPN scale.
        gt_meta: Dict with ``gt_bboxes`` (list of per-image bbox arrays in
            xyxy pixel coords) and ``gt_labels`` (list of per-image label arrays).
        num_classes: Number of object classes.
        strides: Feature map strides per scale (defaults to ``[8, 16, 32]``).
        reg_max: DFL regression max.

    Returns:
        ``(total_loss, {"loss_cls": ..., "loss_reg": ...})``.
    """
    device = preds[0].device
    B = preds[0].shape[0]
    input_h = gt_meta.get("input_h", None)
    if strides is None:
        strides = []
        for p in preds:
            h = p.shape[2]
            if input_h is not None:
                strides.append(input_h // h)
            else:
                strides.append(max(1, round(320 / h)))
    

    cls_sum = torch.tensor(0.0, device=device)
    reg_losses: List[torch.Tensor] = []
    dfl_bins = torch.arange(reg_max + 1, dtype=torch.float32, device=device)
    total_positives = 0
    total_cls_elements = 0

    for scale_idx, pred in enumerate(preds):
        _, C, H, W = pred.shape
        stride = strides[min(scale_idx, len(strides) - 1)]

        cls_pred = pred[:, :num_classes, :, :]
        reg_pred = pred[:, num_classes:, :, :]
        scale_cls_elements = num_classes * H * W

        for b in range(B):
            gt_boxes = gt_meta["gt_bboxes"][b]
            gt_labels = gt_meta["gt_labels"][b]

            gt_boxes_t = torch.as_tensor(gt_boxes, dtype=torch.float32, device=device)
            gt_labels_t = torch.as_tensor(gt_labels, dtype=torch.long, device=device)

            cls_target = torch.zeros(num_classes, H, W, device=device)

            if gt_boxes_t.numel() == 0 or gt_labels_t.numel() == 0:
                cls_sum = cls_sum + F.binary_cross_entropy_with_logits(
                    cls_pred[b], cls_target, reduction="sum"
                )
                total_cls_elements += scale_cls_elements
                continue

            gt_boxes_t = gt_boxes_t.reshape(-1, 4)

            gt_cx = (gt_boxes_t[:, 0] + gt_boxes_t[:, 2]) / 2 / stride
            gt_cy = (gt_boxes_t[:, 1] + gt_boxes_t[:, 3]) / 2 / stride

            gi = gt_cx.long().clamp(0, W - 1)
            gj = gt_cy.long().clamp(0, H - 1)

            n_gt = gt_labels_t.shape[0]
            for k in range(n_gt):
                label = gt_labels_t[k].item()
                if 0 <= label < num_classes:
                    cls_target[label, gj[k], gi[k]] = 1.0

            cls_sum = cls_sum + F.binary_cross_entropy_with_logits(
                cls_pred[b], cls_target, reduction="sum"
            )
            total_cls_elements += scale_cls_elements
            total_positives += n_gt

            for k in range(n_gt):
                j, i = gj[k], gi[k]
                anchor_cx = (i.float() + 0.5) * stride
                anchor_cy = (j.float() + 0.5) * stride

                gt_l = (anchor_cx - gt_boxes_t[k, 0]) / stride
                gt_t = (anchor_cy - gt_boxes_t[k, 1]) / stride
                gt_r = (gt_boxes_t[k, 2] - anchor_cx) / stride
                gt_b = (gt_boxes_t[k, 3] - anchor_cy) / stride
                gt_dist = torch.stack([gt_l, gt_t, gt_r, gt_b]).clamp(
                    0, reg_max - 0.01
                )

                reg_at = reg_pred[b, :, j, i].reshape(4, reg_max + 1)
                pred_dist = F.softmax(reg_at, dim=-1) @ dfl_bins

                reg_losses.append(F.l1_loss(pred_dist, gt_dist))

    loss_cls = cls_sum / max(total_cls_elements, 1)
    loss_reg = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0.0, device=device)

    total = loss_cls + 5.0 * loss_reg

    return total, {
        "loss_cls": loss_cls.detach(),
        "loss_reg": loss_reg.detach(),
    }
