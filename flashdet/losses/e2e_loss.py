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
    img_size: Tuple[int, int] = (640, 640),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute detection loss for a single branch (o2o or o2m).

    Matches Ultralytics v8DetectionLoss:
      - Plain BCE (no focal) for classification
      - Box/L1 losses weighted by target_score per positive
      - All losses normalized by target_scores_sum (sum of soft labels)
      - L1 targets normalized by image size (matches BboxLoss no-DFL path)
    """
    device = cls_preds.device
    B = cls_preds.shape[0]
    total_cls_loss = torch.tensor(0.0, device=device)
    total_box_loss = torch.tensor(0.0, device=device)
    total_l1_loss = torch.tensor(0.0, device=device)
    total_target_scores_sum = 0.0

    for b in range(B):
        cls_pred_b = cls_preds[b]        # [N, num_classes]
        reg_pred_b = reg_preds[b]        # [N, 4]

        decoded_bboxes = _decode_ltrb(anchor_centers, anchor_strides, reg_pred_b)
        cls_scores = cls_pred_b.sigmoid()

        gt_bboxes = gt_bboxes_list[b]
        gt_labels = gt_labels_list[b]

        if gt_bboxes.shape[0] == 0:
            target = torch.zeros_like(cls_pred_b)
            bce = F.binary_cross_entropy_with_logits(cls_pred_b, target, reduction="none")
            total_cls_loss = total_cls_loss + bce.sum()
            continue

        assigned_labels, assigned_bboxes, assigned_scores, fg_mask = assigner.assign(
            anchor_centers, cls_scores, decoded_bboxes, gt_bboxes, gt_labels,
        )

        n_pos = fg_mask.sum().item()
        tss_b = max(assigned_scores.sum().item(), 1.0)
        total_target_scores_sum += tss_b

        # Plain BCE — matches Ultralytics v8DetectionLoss (no focal loss)
        bce = F.binary_cross_entropy_with_logits(
            cls_pred_b, assigned_scores, reduction="none"
        )
        total_cls_loss = total_cls_loss + bce.sum()

        # #region agent log
        if b == 0 and not hasattr(assigner, '_dbg_logged'):
            import os as _os, json as _json, time as _time
            _lp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "debug-387c01.log")
            try:
                with open(_lp, "a") as _f:
                    _f.write(_json.dumps({"sessionId":"387c01","hypothesisId":"B_C_postfix","location":"e2e_loss.py:branch_loss","message":"cls_loss_plain_bce","data":{"branch":"o2o" if assigner.one_to_one else "o2m","n_pos":n_pos,"bce_sum":round(bce.sum().item(),2),"target_scores_sum":round(tss_b,4),"target_score_pos_max":round(assigned_scores[fg_mask].max().item(),4) if n_pos>0 else 0},"timestamp":int(_time.time()*1000)}) + "\n")
            except: pass
        # #endregion

        if n_pos > 0:
            pos_decoded = decoded_bboxes[fg_mask]   # [n_pos, 4]
            pos_target = assigned_bboxes[fg_mask]   # [n_pos, 4]
            pos_reg = reg_pred_b[fg_mask]           # [n_pos, 4]

            # Score-weighted box loss — matches Ultralytics BboxLoss
            score_weight = assigned_scores.sum(-1)[fg_mask].unsqueeze(-1)  # [n_pos, 1]
            ciou = _bbox_iou_aligned(pos_decoded, pos_target)
            box_loss = ((1 - ciou).clamp(min=0).unsqueeze(-1) * score_weight).sum()
            total_box_loss = total_box_loss + box_loss

            # #region agent log
            if b == 0 and not hasattr(assigner, '_dbg_logged'):
                import os as _os2, json as _json2, time as _time2
                _lp2 = _os2.path.join(_os2.path.dirname(_os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__)))), "debug-387c01.log")
                try:
                    with open(_lp2, "a") as _f2:
                        _f2.write(_json2.dumps({"sessionId":"387c01","hypothesisId":"D_postfix","location":"e2e_loss.py:box_loss","message":"box_loss_weighted","data":{"branch":"o2o" if assigner.one_to_one else "o2m","n_pos":n_pos,"ciou_mean":round(ciou.mean().item(),4),"box_loss_weighted":round(box_loss.item(),4),"score_weight_mean":round(score_weight.mean().item(),4)},"timestamp":int(_time2.time()*1000)}) + "\n")
                except: pass
                assigner._dbg_logged = True
            # #endregion

            # Score-weighted L1 loss — matches Ultralytics BboxLoss no-DFL path
            # Reference normalizes by image size: ltrb * stride / imgsz
            pos_centers = anchor_centers[fg_mask]    # [n_pos, 2]
            pos_strides = anchor_strides[fg_mask]    # [n_pos, 1]
            img_h, img_w = img_size
            target_l = (pos_centers[:, 0] - pos_target[:, 0]) / img_w
            target_t = (pos_centers[:, 1] - pos_target[:, 1]) / img_h
            target_r = (pos_target[:, 2] - pos_centers[:, 0]) / img_w
            target_b = (pos_target[:, 3] - pos_centers[:, 1]) / img_h
            target_ltrb = torch.stack([target_l, target_t, target_r, target_b], dim=-1)
            pred_ltrb_pixel = F.softplus(pos_reg, beta=1.0) * pos_strides  # grid → pixel
            pred_l = pred_ltrb_pixel[:, 0] / img_w
            pred_t = pred_ltrb_pixel[:, 1] / img_h
            pred_r = pred_ltrb_pixel[:, 2] / img_w
            pred_b = pred_ltrb_pixel[:, 3] / img_h
            pred_ltrb_norm = torch.stack([pred_l, pred_t, pred_r, pred_b], dim=-1)
            l1_per_pos = F.l1_loss(pred_ltrb_norm, target_ltrb.clamp(min=0), reduction="none").mean(-1, keepdim=True)
            l1_loss = (l1_per_pos * score_weight).sum()
            total_l1_loss = total_l1_loss + l1_loss

    # Normalize by target_scores_sum — matches Ultralytics v8DetectionLoss
    tss = max(total_target_scores_sum, 1.0)
    loss_cls = cls_weight * total_cls_loss / tss
    loss_box = box_weight * total_box_loss / tss
    loss_l1 = l1_weight * total_l1_loss / tss

    total = loss_cls + loss_box + loss_l1

    # #region agent log
    import os as _os3, json as _json3, time as _time3
    _lp3 = _os3.path.join(_os3.path.dirname(_os3.path.dirname(_os3.path.dirname(_os3.path.abspath(__file__)))), "debug-387c01.log")
    if not hasattr(assigner, '_dbg_branch_logged'):
        assigner._dbg_branch_logged = True
        try:
            with open(_lp3, "a") as _f3:
                _f3.write(_json3.dumps({"sessionId":"387c01","hypothesisId":"G_H","location":"e2e_loss.py:branch_summary","message":"loss_components_and_labels","data":{"branch":"o2o" if assigner.one_to_one else "o2m","cls_loss":round(loss_cls.item(),6),"box_loss":round(loss_box.item(),6),"l1_loss":round(loss_l1.item(),6),"l1_raw_sum":round(total_l1_loss.item(),4),"cls_raw_sum":round(total_cls_loss.item(),4),"box_raw_sum":round(total_box_loss.item(),4),"tss":round(tss,4),"total":round(total.item(),4),"cls_weight":cls_weight,"box_weight":box_weight,"l1_weight":l1_weight},"timestamp":int(_time3.time()*1000)}) + "\n")
        except: pass
    # #endregion

    return total, {
        "loss_cls": loss_cls.detach(),
        "loss_box": loss_box.detach(),
        "loss_l1": loss_l1.detach(),
        "num_pos": int(total_target_scores_sum),
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
        self.o2o_assigner = STALAssigner(topk=o2o_topk, strides=strides, one_to_one=True)

    def prog_alpha(self, epoch: int, total_epochs: int) -> float:
        """Compute ProgLoss alpha — per-EPOCH decay matching YOLO26 E2ELoss.

        YOLO26's ``E2ELoss.update()`` is called once per epoch by the
        Ultralytics trainer (trainer.py line 510). The decay function is::

            decay(x) = max(1 - x/(epochs-1), 0) * (o2m_init - final_o2m) + final_o2m

        where ``x`` is the epoch count (0-indexed). With alpha_init=0.8 and
        alpha_final=0.1 the o2m weight decreases linearly from 0.8 → 0.1
        over the full training schedule.
        """
        ratio = min(epoch / max(total_epochs - 1, 1), 1.0)
        return max((1 - ratio) * (self.alpha_init - self.alpha_final) + self.alpha_final, self.alpha_final)

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
        img_size: Optional[Tuple[int, int]] = None,
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
            img_size: (H, W) of input images for L1 normalization.

        Returns:
            total_loss, loss_states dict.
        """
        device = o2o_cls.device
        anchor_centers, anchor_strides = _make_anchor_grid(
            feat_sizes, list(self.strides), device
        )

        # Derive image size from feat_sizes and strides if not provided
        if img_size is None:
            h0, w0 = feat_sizes[0]
            img_size = (h0 * self.strides[0], w0 * self.strides[0])

        alpha = self.prog_alpha(epoch, total_epochs)

        # One-to-Many branch
        o2m_loss, o2m_states = _compute_branch_loss(
            o2m_cls, o2m_reg, anchor_centers, anchor_strides,
            gt_bboxes_list, gt_labels_list, self.num_classes,
            self.o2m_assigner, self.box_weight, self.cls_weight, self.l1_weight,
            img_size=img_size,
        )

        # One-to-One branch
        o2o_loss, o2o_states = _compute_branch_loss(
            o2o_cls, o2o_reg, anchor_centers, anchor_strides,
            gt_bboxes_list, gt_labels_list, self.num_classes,
            self.o2o_assigner, self.box_weight, self.cls_weight, self.l1_weight,
            img_size=img_size,
        )

        # ProgLoss: weighted combination
        B = o2o_cls.shape[0]
        total_per_sample = alpha * o2m_loss + (1 - alpha) * o2o_loss
        # Scale by batch_size — matches Ultralytics v8DetectionLoss.loss()
        # which returns `loss * batch_size`. This ensures gradient magnitude
        # scales properly with batch size (linear scaling rule).
        total = total_per_sample * B

        # #region agent log
        import os as _os, json as _json, time as _time
        _lp = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), "debug-387c01.log")
        if not hasattr(self, '_log_counter'): self._log_counter = 0
        self._log_counter += 1
        if self._log_counter <= 10 or self._log_counter % 50 == 0:
            try:
                with open(_lp, "a") as _f:
                    _f.write(_json.dumps({"sessionId":"387c01","hypothesisId":"F","location":"e2e_loss.py:__call__","message":"batch_size_scaling","data":{"call_count":self._log_counter,"batch_size":B,"total_raw":round(total.item(),4),"total_scaled_by_B":round((total*B).item(),4),"o2m_loss":round(o2m_loss.item(),4),"o2o_loss":round(o2o_loss.item(),4),"o2m_cls":round(o2m_states["loss_cls"].item(),4),"o2m_box":round(o2m_states["loss_box"].item(),4),"o2m_l1":round(o2m_states["loss_l1"].item(),4),"o2o_cls":round(o2o_states["loss_cls"].item(),4),"o2o_box":round(o2o_states["loss_box"].item(),4),"o2o_l1":round(o2o_states["loss_l1"].item(),4),"alpha":round(alpha,4),"epoch":epoch},"timestamp":int(_time.time()*1000)}) + "\n")
            except: pass
        # #endregion

        states = {
            "loss_total": total_per_sample.detach(),
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
