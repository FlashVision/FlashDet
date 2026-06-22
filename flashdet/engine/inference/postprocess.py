"""Shared inference post-processing helpers for various architectures."""

import torch
import torch.nn.functional as F
import torchvision
from typing import List, Tuple


def decode_yolo_predictions(
    preds: List[torch.Tensor],
    num_classes: int,
    input_hw: Tuple[int, int],
    strides: List[int] = None,
    reg_max: int = 16,
    score_thr: float = 0.05,
    nms_thr: float = 0.6,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Decode multi-scale YOLO predictions and apply NMS.

    Args:
        preds: ``[B, num_classes + 4*(reg_max+1), H, W]`` per scale.
        num_classes: Number of object classes.
        input_hw: ``(H, W)`` of the model input image.
        strides: Feature-map strides; defaults to dynamically computed.
        reg_max: DFL regression max.
        score_thr: Confidence threshold for pre-NMS filtering.
        nms_thr: IoU threshold for NMS.

    Returns:
        Per-image list of ``(dets, labels)`` where ``dets`` is
        ``[N, 5]`` (x1, y1, x2, y2, score) and ``labels`` is ``[N]``.
    """
    device = preds[0].device
    if strides is None:
        strides = [input_hw[0] // p.shape[2] for p in preds]
    B = preds[0].shape[0]
    dfl_bins = torch.arange(reg_max + 1, dtype=torch.float32, device=device)

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []

    for scale_idx, pred in enumerate(preds):
        _, C, H, W = pred.shape
        stride = strides[min(scale_idx, len(strides) - 1)]

        cls_pred = pred[:, :num_classes, :, :]
        reg_pred = pred[:, num_classes:, :, :]

        yv, xv = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        anchor_x = (xv.reshape(-1) + 0.5) * stride
        anchor_y = (yv.reshape(-1) + 0.5) * stride

        cls_flat = cls_pred.reshape(B, num_classes, -1).permute(0, 2, 1).sigmoid()
        reg_flat = reg_pred.reshape(B, 4, reg_max + 1, -1)
        dist = F.softmax(reg_flat, dim=2)
        dist = (dist * dfl_bins.view(1, 1, -1, 1)).sum(dim=2) * stride

        x1 = anchor_x.unsqueeze(0) - dist[:, 0]
        y1 = anchor_y.unsqueeze(0) - dist[:, 1]
        x2 = anchor_x.unsqueeze(0) + dist[:, 2]
        y2 = anchor_y.unsqueeze(0) + dist[:, 3]
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        all_boxes.append(boxes)
        all_scores.append(cls_flat)

    cat_boxes = torch.cat(all_boxes, dim=1)
    cat_scores = torch.cat(all_scores, dim=1)

    results: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for b in range(B):
        scores_b = cat_scores[b]
        boxes_b = cat_boxes[b]

        max_scores, max_labels = scores_b.max(dim=-1)
        keep_mask = max_scores > score_thr
        if keep_mask.sum() == 0:
            results.append((
                torch.zeros(0, 5, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            ))
            continue

        filt_boxes = boxes_b[keep_mask]
        filt_scores = max_scores[keep_mask]
        filt_labels = max_labels[keep_mask]

        nms_idx = torchvision.ops.nms(filt_boxes, filt_scores, nms_thr)
        dets = torch.cat([filt_boxes[nms_idx], filt_scores[nms_idx].unsqueeze(-1)], dim=-1)
        results.append((dets, filt_labels[nms_idx]))

    return results


def decode_detr_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    img_hw: Tuple[int, int],
    score_thr: float = 0.05,
    use_softmax: bool = True,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Decode DETR/RT-DETR predictions into (dets, labels) per image.

    Args:
        pred_logits: [B, num_queries, num_classes(+1)].
        pred_boxes: [B, num_queries, 4] in cxcywh normalized format.
        img_hw: (H, W) of the input image.
        score_thr: Confidence threshold.
        use_softmax: If True, use softmax (DETR with background class);
            if False, use sigmoid (RT-DETR).

    Returns:
        Per-image list of ``(dets, labels)``.
    """
    B = pred_logits.shape[0]
    h, w = img_hw
    results = []
    for b in range(B):
        if use_softmax:
            probs = pred_logits[b].softmax(-1)
            scores, labels = probs[:, :-1].max(-1)
        else:
            scores_all = pred_logits[b].sigmoid()
            scores, labels = scores_all.max(-1)

        keep = scores > score_thr
        boxes = pred_boxes[b, keep]
        boxes_xyxy = torch.stack([
            (boxes[:, 0] - boxes[:, 2] / 2) * w,
            (boxes[:, 1] - boxes[:, 3] / 2) * h,
            (boxes[:, 0] + boxes[:, 2] / 2) * w,
            (boxes[:, 1] + boxes[:, 3] / 2) * h,
        ], dim=-1)
        dets = torch.cat([boxes_xyxy, scores[keep].unsqueeze(-1)], dim=-1)
        results.append((dets, labels[keep]))
    return results
