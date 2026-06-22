"""
Grounding DINO — Open-Vocabulary Object Detection.

Text-conditioned detection using a CLIP text encoder for text features and
a visual backbone, fused through cross-modality attention for open-set
detection.

Reference:
    Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training
    for Open-Set Object Detection", 2023.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.registry import DETECTORS
from flashdet.models.backbone.resnet import ResNetBackbone
from flashdet.models.backbone.text_encoder import TextEncoder
from flashdet.models.transformer.vision_language_fusion import VisionLanguageFusion
from flashdet.models.head.grounding_dino_decoder import GroundingDINODecoder

logger = logging.getLogger(__name__)


@DETECTORS.register("GroundingDINO")
class GroundingDINO(nn.Module):
    """Grounding DINO: open-vocabulary object detector.

    Combines a vision backbone with a text encoder and fuses them
    through bi-directional cross-attention for text-conditioned
    detection.

    Args:
        num_queries: Number of detection queries.
        d_model: Hidden dimension for transformer.
        nhead: Number of attention heads.
        num_encoder_layers: Number of vision-language fusion layers.
        num_decoder_layers: Decoder depth.
        backbone: ResNet variant for visual backbone.
        pretrained_backbone: Load ImageNet-pretrained backbone.
        vocab_size: Text encoder vocabulary size.
        max_text_len: Maximum text token length.
        text_encoder_depth: Depth of text encoder transformer.
    """

    def __init__(
        self,
        num_queries: int = 900,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 6,
        backbone: str = "resnet50",
        pretrained_backbone: bool = True,
        vocab_size: int = 49408,
        max_text_len: int = 77,
        text_encoder_depth: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        self.visual_backbone = ResNetBackbone(backbone, d_model, pretrained_backbone)

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, embed_dim=d_model,
            max_len=max_text_len, depth=text_encoder_depth, nhead=nhead,
        )

        self.fusion_layers = nn.ModuleList([
            VisionLanguageFusion(d_model, nhead) for _ in range(num_encoder_layers)
        ])

        self.decoder = GroundingDINODecoder(
            d_model=d_model, nhead=nhead,
            num_layers=num_decoder_layers, num_queries=num_queries,
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        gt_meta: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        vis_feat = self.visual_backbone(images)
        B, C, H, W = vis_feat.shape
        vis_tokens = vis_feat.flatten(2).permute(0, 2, 1)

        if input_ids is None:
            input_ids = torch.zeros(B, 10, dtype=torch.long, device=images.device)
            attention_mask = torch.ones(B, 10, dtype=torch.long, device=images.device)

        text_feat = self.text_encoder(input_ids, attention_mask)

        for fusion in self.fusion_layers:
            vis_tokens, text_feat = fusion(vis_tokens, text_feat, attention_mask)

        dec_out = self.decoder(vis_tokens, text_feat, attention_mask)

        if self.training and gt_meta is not None:
            loss, loss_states = self._compute_loss(dec_out, gt_meta, images.shape[2:])
            return {"loss": loss, "loss_states": loss_states}

        return {"preds": dec_out}

    def _compute_loss(self, dec_out: Dict, gt_meta: Dict, img_shape: Tuple[int, int]) -> Tuple[torch.Tensor, Dict]:
        pred_logits = dec_out["pred_logits"]
        pred_boxes = dec_out["pred_boxes"]
        device = pred_logits.device
        B, Q, T = pred_logits.shape

        total_cls, total_l1, total_giou = [], [], []

        for b in range(B):
            gt_b = torch.as_tensor(gt_meta["gt_bboxes"][b], dtype=torch.float32, device=device)
            gt_l = torch.as_tensor(gt_meta["gt_labels"][b], dtype=torch.long, device=device)

            if gt_b.numel() == 0:
                total_cls.append(pred_logits[b].sum() * 0)
                continue

            h, w = img_shape
            gt_cx = (gt_b[:, 0] + gt_b[:, 2]) / 2 / w
            gt_cy = (gt_b[:, 1] + gt_b[:, 3]) / 2 / h
            gt_w = (gt_b[:, 2] - gt_b[:, 0]) / w
            gt_h = (gt_b[:, 3] - gt_b[:, 1]) / h
            gt_norm = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=-1)

            cost_l1 = torch.cdist(pred_boxes[b], gt_norm, p=1)
            max_logit = pred_logits[b].max(dim=-1).values
            cost_cls = -max_logit.unsqueeze(1).expand(-1, gt_l.shape[0])
            cost = cost_cls + 5.0 * cost_l1

            from scipy.optimize import linear_sum_assignment
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            row = torch.as_tensor(row, dtype=torch.long, device=device)
            col = torch.as_tensor(col, dtype=torch.long, device=device)

            matched_boxes = pred_boxes[b, row]
            target_boxes = gt_norm[col]
            total_l1.append(F.l1_loss(matched_boxes, target_boxes))

            from flashdet.models.assignment.hungarian_matcher import cxcywh_to_xyxy
            s_xy = cxcywh_to_xyxy(matched_boxes)
            t_xy = cxcywh_to_xyxy(target_boxes)
            lt = torch.max(s_xy[:, :2], t_xy[:, :2])
            rb = torch.min(s_xy[:, 2:], t_xy[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, 0] * wh[:, 1]
            a1 = (s_xy[:, 2] - s_xy[:, 0]) * (s_xy[:, 3] - s_xy[:, 1])
            a2 = (t_xy[:, 2] - t_xy[:, 0]) * (t_xy[:, 3] - t_xy[:, 1])
            union = a1 + a2 - inter
            iou = inter / union.clamp(min=1e-6)
            enc_lt = torch.min(s_xy[:, :2], t_xy[:, :2])
            enc_rb = torch.max(s_xy[:, 2:], t_xy[:, 2:])
            enc_wh = (enc_rb - enc_lt).clamp(min=0)
            enc_area = enc_wh[:, 0] * enc_wh[:, 1]
            giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
            total_giou.append((1 - giou).mean())

            target_per_query = torch.zeros(Q, T, dtype=torch.float32, device=device)
            matched_labels = gt_l[col]
            safe_labels = matched_labels.clamp(max=T - 1)
            target_per_query[row, safe_labels] = 1.0
            total_cls.append(F.binary_cross_entropy_with_logits(
                pred_logits[b], target_per_query,
                pos_weight=torch.tensor(5.0, device=device),
            ))

        loss_cls = torch.stack(total_cls).mean() if total_cls else pred_logits.sum() * 0
        loss_l1 = torch.stack(total_l1).mean() if total_l1 else pred_boxes.sum() * 0
        loss_giou = torch.stack(total_giou).mean() if total_giou else pred_boxes.sum() * 0

        total = loss_cls + 5.0 * loss_l1 + 2.0 * loss_giou
        return total, {"loss_cls": loss_cls.detach(), "loss_l1": loss_l1.detach(), "loss_giou": loss_giou.detach()}

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        score_thr: float = 0.3,
        nms_thr: float = 0.5,
    ) -> List:
        self.eval()
        if input_ids is None:
            B = images.shape[0]
            input_ids = torch.randint(1, 100, (B, 10), device=images.device)
            attention_mask = torch.ones(B, 10, dtype=torch.long, device=images.device)

        out = self.forward(images, input_ids, attention_mask)
        dec_out = out["preds"]
        pred_logits = dec_out["pred_logits"]
        pred_boxes = dec_out["pred_boxes"]
        B = pred_logits.shape[0]
        h, w = images.shape[2:]

        results = []
        for b in range(B):
            scores_per_query = pred_logits[b].sigmoid().max(dim=-1)
            keep = scores_per_query.values > score_thr
            if keep.sum() == 0:
                dets = torch.zeros(0, 5, device=images.device)
                labels = torch.zeros(0, dtype=torch.long, device=images.device)
                results.append((dets, labels))
                continue

            boxes = pred_boxes[b, keep]
            boxes_xyxy = torch.stack([
                (boxes[:, 0] - boxes[:, 2] / 2).clamp(0) * w,
                (boxes[:, 1] - boxes[:, 3] / 2).clamp(0) * h,
                (boxes[:, 0] + boxes[:, 2] / 2).clamp(max=1) * w,
                (boxes[:, 1] + boxes[:, 3] / 2).clamp(max=1) * h,
            ], dim=-1)
            conf = scores_per_query.values[keep]
            labels = scores_per_query.indices[keep]

            from torchvision.ops import nms
            nms_keep = nms(boxes_xyxy, conf, nms_thr)
            dets = torch.cat([boxes_xyxy[nms_keep], conf[nms_keep].unsqueeze(-1)], dim=-1)
            results.append((dets, labels[nms_keep]))
        return results

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "GroundingDINO",
            "num_queries": self.num_queries,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
