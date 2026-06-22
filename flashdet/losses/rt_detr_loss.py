"""RT-DETR loss with focal classification and GIoU regression."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from flashdet.models.assignment.hungarian_matcher import cxcywh_to_xyxy


def compute_rt_detr_loss(
    dec_out: Dict,
    gt_meta: Dict,
    num_classes: int,
    num_queries: int,
    img_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute RT-DETR loss.

    Args:
        dec_out: Decoder output dict with 'pred_logits' and 'pred_boxes'.
        gt_meta: Dict with gt_bboxes (xyxy pixel) and gt_labels.
        num_classes: Number of object classes.
        num_queries: Number of selected object queries.
        img_shape: (H, W) of the input image.

    Returns:
        (total_loss, loss_states dict).
    """
    pred_logits = dec_out["pred_logits"]
    pred_boxes = dec_out["pred_boxes"]
    device = pred_logits.device
    B = pred_logits.shape[0]

    gt_labels_list, gt_boxes_list = [], []
    for i in range(B):
        gt_b = torch.as_tensor(gt_meta["gt_bboxes"][i], dtype=torch.float32, device=device)
        gt_l = torch.as_tensor(gt_meta["gt_labels"][i], dtype=torch.long, device=device)
        if gt_b.numel() > 0:
            cx = (gt_b[:, 0] + gt_b[:, 2]) / 2 / img_shape[1]
            cy = (gt_b[:, 1] + gt_b[:, 3]) / 2 / img_shape[0]
            w = (gt_b[:, 2] - gt_b[:, 0]) / img_shape[1]
            h = (gt_b[:, 3] - gt_b[:, 1]) / img_shape[0]
            gt_boxes_list.append(torch.stack([cx, cy, w, h], dim=-1))
        else:
            gt_boxes_list.append(gt_b.reshape(0, 4))
        gt_labels_list.append(gt_l)

    indices = _match(pred_logits, pred_boxes, gt_labels_list, gt_boxes_list)

    target_classes = torch.full(
        (B, num_queries), num_classes, dtype=torch.long, device=device,
    )
    for b, (si, ti) in enumerate(indices):
        if si.numel() > 0:
            target_classes[b, si] = gt_labels_list[b][ti]

    target_onehot = F.one_hot(target_classes, num_classes + 1)[..., :-1].float()
    loss_cls = _sigmoid_focal_loss(pred_logits, target_onehot, alpha=0.25, gamma=2.0)

    src_boxes, tgt_boxes = [], []
    for b, (si, ti) in enumerate(indices):
        if si.numel() > 0:
            src_boxes.append(pred_boxes[b, si])
            tgt_boxes.append(gt_boxes_list[b][ti])

    if src_boxes:
        src_cat = torch.cat(src_boxes)
        tgt_cat = torch.cat(tgt_boxes)
        loss_l1 = F.l1_loss(src_cat, tgt_cat, reduction="mean")
        loss_giou = _giou_loss(src_cat, tgt_cat)
    else:
        loss_l1 = pred_boxes.sum() * 0
        loss_giou = pred_boxes.sum() * 0

    total = loss_cls + 5.0 * loss_l1 + 2.0 * loss_giou

    return total, {
        "loss_cls": loss_cls.detach(),
        "loss_l1": loss_l1.detach(),
        "loss_giou": loss_giou.detach(),
    }


def _sigmoid_focal_loss(
    inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0,
) -> torch.Tensor:
    prob = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def _giou_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    s, t = cxcywh_to_xyxy(src), cxcywh_to_xyxy(tgt)
    lt = torch.max(s[:, :2], t[:, :2])
    rb = torch.min(s[:, 2:], t[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    a1 = (s[:, 2] - s[:, 0]) * (s[:, 3] - s[:, 1])
    a2 = (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    union = a1 + a2 - inter
    iou = inter / union.clamp(min=1e-6)
    enc_lt = torch.min(s[:, :2], t[:, :2])
    enc_rb = torch.max(s[:, 2:], t[:, 2:])
    enc_area = (enc_rb[:, 0] - enc_lt[:, 0]).clamp(min=0) * (enc_rb[:, 1] - enc_lt[:, 1]).clamp(min=0)
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return (1 - giou).mean()


@torch.no_grad()
def _match(pred_logits, pred_boxes, gt_labels_list, gt_boxes_list):
    B = pred_logits.shape[0]
    indices = []
    for b in range(B):
        if gt_labels_list[b].numel() == 0:
            dev = pred_logits.device
            indices.append((torch.tensor([], dtype=torch.long, device=dev),
                            torch.tensor([], dtype=torch.long, device=dev)))
            continue
        prob = pred_logits[b].sigmoid()
        cost_cls = -prob[:, gt_labels_list[b]]
        cost_l1 = torch.cdist(pred_boxes[b], gt_boxes_list[b], p=1)

        s_xy = cxcywh_to_xyxy(pred_boxes[b])
        t_xy = cxcywh_to_xyxy(gt_boxes_list[b])
        lt = torch.max(s_xy[:, None, :2], t_xy[None, :, :2])
        rb = torch.min(s_xy[:, None, 2:], t_xy[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        a1 = (s_xy[:, 2] - s_xy[:, 0]) * (s_xy[:, 3] - s_xy[:, 1])
        a2 = (t_xy[:, 2] - t_xy[:, 0]) * (t_xy[:, 3] - t_xy[:, 1])
        union = a1[:, None] + a2[None, :] - inter
        iou = inter / union.clamp(min=1e-6)
        enc_lt = torch.min(s_xy[:, None, :2], t_xy[None, :, :2])
        enc_rb = torch.max(s_xy[:, None, 2:], t_xy[None, :, 2:])
        enc_area = (enc_rb[..., 0] - enc_lt[..., 0]).clamp(0) * (enc_rb[..., 1] - enc_lt[..., 1]).clamp(0)
        giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
        cost_giou = -giou

        cost = cost_cls + 5.0 * cost_l1 + 2.0 * cost_giou
        row, col = linear_sum_assignment(cost.cpu().numpy())
        dev = pred_logits.device
        indices.append((torch.as_tensor(row, dtype=torch.long, device=dev),
                        torch.as_tensor(col, dtype=torch.long, device=dev)))
    return indices
