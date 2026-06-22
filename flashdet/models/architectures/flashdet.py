"""
FlashDet — YOLO26-Based Lightweight Object Detector.

Architecture built on YOLO26 principles:
  - YOLO11 Backbone (C3k2 + SPPF + C2PSA)
  - YOLO11 PAN-FPN Neck
  - DFL-free Dual Detection Head (One-to-One + One-to-Many)
  - STAL (Small-Target-Aware Label Assignment)
  - ProgLoss (Progressive Loss shifting from o2m → o2o)
  - MuSGD optimizer (Muon + SGD hybrid)

Training Pipeline:
    Dataset → Augmentation → FlashDet (YOLO26) →
      ├── Classification Loss (BCE)
      ├── Box Loss (CIoU + L1, ProgLoss weighted)
      └── STAL Assignment
          → MuSGD → Updated Weights

Model Sizes:
    - FlashDet-N  (width=0.25, depth=0.33): ~1.5M params
    - FlashDet-S  (width=0.50, depth=0.33): ~5.4M params
    - FlashDet-M  (width=1.00, depth=0.67): ~18M params

Reference:
    Ultralytics YOLO26 (2026).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from flashdet.registry import DETECTORS
from flashdet.models.backbone.yolov11_backbone import YOLOv11Backbone
from flashdet.models.neck.yolov11_neck import YOLOv11Neck
from flashdet.models.head.e2e_head import E2EDualHead
from flashdet.losses.e2e_loss import E2EDetectionLoss, _make_anchor_grid, _decode_ltrb

logger = logging.getLogger(__name__)

SIZE_CONFIGS = {
    "n": {"width_mult": 0.25, "depth_mult": 0.33, "use_c2psa": True},
    "s": {"width_mult": 0.50, "depth_mult": 0.33, "use_c2psa": True},
    "m": {"width_mult": 1.00, "depth_mult": 0.67, "use_c2psa": True},
    "l": {"width_mult": 1.25, "depth_mult": 1.00, "use_c2psa": True},
    "x": {"width_mult": 1.50, "depth_mult": 1.00, "use_c2psa": True},
}


@DETECTORS.register("FlashDet")
class FlashDet(nn.Module):
    """FlashDet — YOLO26-based lightweight object detector.

    Args:
        num_classes: Number of detection classes.
        size: Model size variant ("n", "s", "m", "l", "x").
        width_mult: Override width multiplier (ignored if size is set).
        depth_mult: Override depth multiplier (ignored if size is set).
        use_c2psa: Use C2PSA attention in backbone.
        strides: Feature pyramid strides.
        total_epochs: Total training epochs (for ProgLoss scheduling).
        prog_alpha_init: ProgLoss initial o2m weight.
        prog_alpha_final: ProgLoss final o2m weight.
        o2m_topk: One-to-many assigner top-k.
        o2o_topk: One-to-one assigner top-k.
        box_weight: CIoU loss weight.
        cls_weight: Classification loss weight.
        l1_weight: L1 distance loss weight.
    """

    def __init__(
        self,
        num_classes: int = 80,
        size: str = "n",
        width_mult: Optional[float] = None,
        depth_mult: Optional[float] = None,
        use_c2psa: Optional[bool] = None,
        strides: Tuple[int, ...] = (8, 16, 32),
        total_epochs: int = 100,
        prog_alpha_init: float = 1.0,
        prog_alpha_final: float = 0.0,
        o2m_topk: int = 10,
        o2o_topk: int = 7,
        box_weight: float = 7.5,
        cls_weight: float = 1.0,
        l1_weight: float = 0.5,
    ):
        super().__init__()

        cfg = SIZE_CONFIGS.get(size, SIZE_CONFIGS["n"])
        wm = width_mult if width_mult is not None else cfg["width_mult"]
        dm = depth_mult if depth_mult is not None else cfg["depth_mult"]
        c2psa = use_c2psa if use_c2psa is not None else cfg["use_c2psa"]

        self.num_classes = num_classes
        self.size = size
        self.strides = strides
        self.total_epochs = total_epochs

        # Backbone: YOLO11 C3k2 + SPPF + C2PSA
        self.backbone = YOLOv11Backbone(wm, dm, c2psa)

        # Neck: YOLO11 PAN-FPN
        neck_out = self.backbone.out_channels[1]
        self.neck = YOLOv11Neck(self.backbone.out_channels, neck_out)

        # Head: DFL-free dual detection head (o2o + o2m)
        self.head = E2EDualHead(num_classes, neck_out, num_levels=3)

        # Loss: E2E with ProgLoss + STAL
        self.loss_fn = E2EDetectionLoss(
            num_classes=num_classes,
            strides=strides,
            alpha_init=prog_alpha_init,
            alpha_final=prog_alpha_final,
            o2m_topk=o2m_topk,
            o2o_topk=o2o_topk,
            box_weight=box_weight,
            cls_weight=cls_weight,
            l1_weight=l1_weight,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Initialize cls bias for low initial false-positive rate
        for heads in [self.head.o2o_heads, self.head.o2m_heads]:
            for h in heads:
                nn.init.constant_(h.cls_pred.bias, -4.595)

    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Optional[Dict] = None,
        epoch: int = 0,
        compute_loss: bool = False,
        return_features: bool = False,
        **kwargs,
    ) -> Dict:
        """Forward pass.

        Args:
            x: [B, 3, H, W] input tensor.
            gt_meta: Ground truth with "gt_bboxes" and "gt_labels".
            epoch: Current training epoch (for ProgLoss schedule).
            compute_loss: Force loss computation even in eval mode.
            return_features: Include backbone/neck features in output.

        Returns:
            Training: {"loss", "loss_states"} and optionally features.
            Inference: {"preds"} — o2o head outputs for NMS-free decode.
        """
        features = self.backbone(x)
        neck_feats = self.neck(features)
        head_out = self.head(neck_feats, training=self.training or compute_loss)

        result: Dict = {}

        if return_features:
            result["backbone_features"] = features
            result["fpn_features"] = neck_feats

        if (self.training or compute_loss) and gt_meta is not None:
            device = x.device
            gt_bboxes_list = []
            gt_labels_list = []
            for b_bboxes, b_labels in zip(gt_meta["gt_bboxes"], gt_meta["gt_labels"]):
                gt_bboxes_list.append(torch.as_tensor(b_bboxes, dtype=torch.float32, device=device).reshape(-1, 4))
                gt_labels_list.append(torch.as_tensor(b_labels, dtype=torch.long, device=device).reshape(-1))

            loss, loss_states = self.loss_fn(
                o2o_cls=head_out["o2o_cls"],
                o2o_reg=head_out["o2o_reg"],
                o2m_cls=head_out["o2m_cls"],
                o2m_reg=head_out["o2m_reg"],
                gt_bboxes_list=gt_bboxes_list,
                gt_labels_list=gt_labels_list,
                feat_sizes=head_out["feat_sizes"],
                epoch=epoch,
                total_epochs=self.total_epochs,
            )

            result["loss"] = loss
            result["loss_states"] = loss_states
            result["preds"] = head_out["o2o_cls"]  # for compatibility
            result["o2o_cls"] = head_out["o2o_cls"]
            result["o2o_reg"] = head_out["o2o_reg"]
            result["feat_sizes"] = head_out["feat_sizes"]
        else:
            result["preds"] = head_out["o2o_cls"]
            result["o2o_cls"] = head_out["o2o_cls"]
            result["o2o_reg"] = head_out["o2o_reg"]
            result["feat_sizes"] = head_out["feat_sizes"]

        return result

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        img_metas: Optional[Dict] = None,
        score_thr: float = 0.25,
        nms_thr: float = 0.7,
        max_det: int = 300,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """NMS-free inference using the one-to-one head.

        For FlashDet, the o2o head produces one prediction per anchor.
        We decode, threshold, and optionally apply lightweight NMS for safety.

        Returns:
            List of (det_bboxes [N,5], det_labels [N]) per image.
        """
        self.eval()
        out = self.forward(x)
        cls_logits = out["o2o_cls"]   # [B, N, num_classes]
        reg_preds = out["o2o_reg"]    # [B, N, 4]
        feat_sizes = out["feat_sizes"]

        device = x.device
        anchor_centers, anchor_strides = _make_anchor_grid(
            feat_sizes, list(self.strides), device
        )

        results = []
        B = cls_logits.shape[0]

        H, W = x.shape[2], x.shape[3]
        for b in range(B):
            scores = cls_logits[b].sigmoid()  # [N, num_classes]
            boxes = _decode_ltrb(anchor_centers, anchor_strides, reg_preds[b])  # [N, 4]
            boxes[:, 0].clamp_(min=0, max=W)
            boxes[:, 1].clamp_(min=0, max=H)
            boxes[:, 2].clamp_(min=0, max=W)
            boxes[:, 3].clamp_(min=0, max=H)

            max_scores, labels = scores.max(dim=1)  # [N], [N]

            valid_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 1.0
            keep = (max_scores > score_thr) & valid_area
            if keep.sum() == 0:
                results.append((
                    torch.zeros((0, 5), device=device),
                    torch.zeros((0,), dtype=torch.long, device=device),
                ))
                continue

            scores_k = max_scores[keep]
            boxes_k = boxes[keep]
            labels_k = labels[keep]

            # Lightweight NMS (safety net, o2o head is already near-duplicate-free)
            nms_idx = nms(boxes_k, scores_k, nms_thr)
            nms_idx = nms_idx[:max_det]

            det_boxes = boxes_k[nms_idx]
            det_scores = scores_k[nms_idx]
            det_labels = labels_k[nms_idx]

            det_bboxes = torch.cat([det_boxes, det_scores[:, None]], dim=1)
            results.append((det_bboxes, det_labels))

        return results

    def get_model_info(self) -> Dict:
        """Get model information."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Inference uses only o2o heads (o2m is training-only)
        o2m_params = sum(p.numel() for p in self.head.o2m_heads.parameters())
        inference_params = total - o2m_params

        return {
            "name": f"FlashDet-{self.size.upper()}",
            "num_classes": self.num_classes,
            "size": self.size,
            "total_params": total,
            "trainable_params": trainable,
            "inference_params": inference_params,
            "params_mb": total * 4 / (1024 ** 2),
            "inference_params_mb": inference_params * 4 / (1024 ** 2),
            "inference_fp16_mb": inference_params * 2 / (1024 ** 2),
        }
