"""
DETR — DEtection TRansformer.

End-to-end object detection with a transformer encoder-decoder architecture
and Hungarian (bipartite) matching for set-based loss.

Architecture:
    ResNet backbone → positional encoding → Transformer encoder-decoder
    → FFN prediction heads (class + bbox per query)

Reference:
    Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from flashdet.registry import BACKBONES

logger = logging.getLogger(__name__)


class PositionalEncoding2D(nn.Module):
    """Fixed 2-D sinusoidal positional encoding for spatial feature maps."""

    def __init__(self, d_model: int = 256, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        half = self.d_model // 2
        device = x.device

        y_pos = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1).expand(H, W)
        x_pos = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(0).expand(H, W)

        dim_t = torch.arange(half, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / half)

        pos_x = x_pos.unsqueeze(-1) / dim_t
        pos_y = y_pos.unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)
        return pos.unsqueeze(0).expand(B, -1, -1, -1)


class ResNetBackbone(nn.Module):
    """ResNet backbone producing a single feature map for DETR.

    Uses torchvision ResNet and returns the output of layer4 (stride 32).
    A 1x1 conv projects the channels to the transformer hidden dimension.
    """

    def __init__(self, variant: str = "resnet50", d_model: int = 256, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tv_models

        factory = {
            "resnet18": (tv_models.resnet18, 512),
            "resnet34": (tv_models.resnet34, 512),
            "resnet50": (tv_models.resnet50, 2048),
            "resnet101": (tv_models.resnet101, 2048),
        }
        if variant not in factory:
            raise ValueError(f"Unknown ResNet variant '{variant}'. Choose from {list(factory.keys())}")

        builder, out_ch = factory[variant]
        weights = "DEFAULT" if pretrained else None
        resnet = builder(weights=weights)

        self.body = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.proj = nn.Conv2d(out_ch, d_model, 1)
        self.out_channels = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.body(x))


class DETRTransformer(nn.Module):
    """Standard Transformer encoder-decoder for DETR."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(
        self,
        src: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H, W = src.shape
        src_flat = src.flatten(2).permute(0, 2, 1)
        pos_flat = pos_embed.flatten(2).permute(0, 2, 1)

        memory = self.encoder(src_flat + pos_flat)

        queries = query_embed.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)
        return out


class HungarianMatcher(nn.Module):
    """Bipartite matching between predictions and ground-truth using the
    Hungarian algorithm, as described in the DETR paper."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_labels_list: List[torch.Tensor],
        gt_boxes_list: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        B, N, C = pred_logits.shape
        indices = []

        for b in range(B):
            if gt_labels_list[b].numel() == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_logits.device),
                    torch.tensor([], dtype=torch.long, device=pred_logits.device),
                ))
                continue

            prob = pred_logits[b].softmax(-1)
            cost_cls = -prob[:, gt_labels_list[b]]

            cost_box = torch.cdist(pred_boxes[b], gt_boxes_list[b], p=1)

            cost_giou = -self._generalized_iou(
                self._cxcywh_to_xyxy(pred_boxes[b]),
                self._cxcywh_to_xyxy(gt_boxes_list[b]),
            )

            cost = self.cost_class * cost_cls + self.cost_bbox * cost_box + self.cost_giou * cost_giou
            row, col = linear_sum_assignment(cost.cpu().numpy())
            indices.append((
                torch.as_tensor(row, dtype=torch.long, device=pred_logits.device),
                torch.as_tensor(col, dtype=torch.long, device=pred_logits.device),
            ))
        return indices

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    @staticmethod
    def _generalized_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Pairwise GIoU between two sets of boxes (xyxy format)."""
        N, M = boxes1.shape[0], boxes2.shape[0]

        lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]

        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / union.clamp(min=1e-6)

        enc_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
        enc_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
        enc_wh = (enc_rb - enc_lt).clamp(min=0)
        enc_area = enc_wh[..., 0] * enc_wh[..., 1]

        return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


@BACKBONES.register("DETR")
class DETR(nn.Module):
    """DETR: End-to-End Object Detection with Transformers.

    Args:
        num_classes: Number of detection classes (excluding background).
        num_queries: Number of object queries.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Transformer encoder depth.
        num_decoder_layers: Transformer decoder depth.
        dim_feedforward: FFN hidden size.
        dropout: Dropout rate.
        backbone: ResNet variant for the backbone.
        pretrained_backbone: Load ImageNet-pretrained backbone.
        cost_class: Hungarian matching class cost weight.
        cost_bbox: Hungarian matching L1 bbox cost weight.
        cost_giou: Hungarian matching GIoU cost weight.
        loss_ce_weight: Classification loss weight.
        loss_bbox_weight: L1 bbox regression loss weight.
        loss_giou_weight: GIoU loss weight.
    """

    def __init__(
        self,
        num_classes: int = 80,
        num_queries: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        backbone: str = "resnet50",
        pretrained_backbone: bool = True,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_ce_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = ResNetBackbone(backbone, d_model, pretrained_backbone)
        self.pos_encoder = PositionalEncoding2D(d_model)
        self.transformer = DETRTransformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.loss_ce_weight = loss_ce_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight

        self._init_weights()

    def _init_weights(self):
        for p in self.class_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.bbox_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.uniform_(self.query_embed.weight)

    def forward(
        self,
        x: torch.Tensor,
        gt_meta: Optional[Dict] = None,
        **kwargs,
    ) -> Dict:
        features = self.backbone(x)
        pos = self.pos_encoder(features)

        hs = self.transformer(features, self.query_embed.weight, pos)

        pred_logits = self.class_head(hs)
        pred_boxes = self.bbox_head(hs).sigmoid()

        if self.training and gt_meta is not None:
            loss, loss_states = self._compute_loss(pred_logits, pred_boxes, gt_meta)
            return {"loss": loss, "loss_states": loss_states}

        return {"preds": {"logits": pred_logits, "boxes": pred_boxes}}

    def _compute_loss(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_meta: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        device = pred_logits.device
        B = pred_logits.shape[0]

        gt_labels_list = []
        gt_boxes_list = []
        for i in range(B):
            gt_b = torch.as_tensor(gt_meta["gt_bboxes"][i], dtype=torch.float32, device=device)
            gt_l = torch.as_tensor(gt_meta["gt_labels"][i], dtype=torch.long, device=device)
            if gt_b.numel() > 0:
                gt_boxes_list.append(self._xyxy_to_cxcywh_norm(gt_b, gt_meta["img"].shape[2:]))
            else:
                gt_boxes_list.append(gt_b.reshape(0, 4))
            gt_labels_list.append(gt_l)

        indices = self.matcher(pred_logits, pred_boxes, gt_labels_list, gt_boxes_list)

        # Classification loss with "no-object" class
        target_classes = torch.full(
            (B, self.num_queries), self.num_classes, dtype=torch.long, device=device,
        )
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                target_classes[b, src_idx] = gt_labels_list[b][tgt_idx]

        eos_weight = torch.ones(self.num_classes + 1, device=device)
        eos_weight[-1] = 0.1
        loss_ce = F.cross_entropy(
            pred_logits.flatten(0, 1), target_classes.flatten(), weight=eos_weight,
        )

        # Bbox losses (L1 + GIoU) on matched pairs only
        src_boxes_all = []
        tgt_boxes_all = []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() > 0:
                src_boxes_all.append(pred_boxes[b, src_idx])
                tgt_boxes_all.append(gt_boxes_list[b][tgt_idx])

        if src_boxes_all:
            src_cat = torch.cat(src_boxes_all)
            tgt_cat = torch.cat(tgt_boxes_all)
            loss_bbox = F.l1_loss(src_cat, tgt_cat, reduction="mean")
            loss_giou = (1 - torch.diag(
                HungarianMatcher._generalized_iou(
                    HungarianMatcher._cxcywh_to_xyxy(src_cat),
                    HungarianMatcher._cxcywh_to_xyxy(tgt_cat),
                )
            )).mean()
        else:
            loss_bbox = pred_boxes.sum() * 0
            loss_giou = pred_boxes.sum() * 0

        total = (
            self.loss_ce_weight * loss_ce
            + self.loss_bbox_weight * loss_bbox
            + self.loss_giou_weight * loss_giou
        )

        return total, {
            "loss_ce": loss_ce.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_giou": loss_giou.detach(),
        }

    @staticmethod
    def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_shape: Tuple[int, int]) -> torch.Tensor:
        h, w = img_shape
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return torch.stack([cx, cy, bw, bh], dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, score_thr: float = 0.5) -> List[Dict]:
        self.eval()
        out = self.forward(x)
        pred_logits = out["preds"]["logits"]
        pred_boxes = out["preds"]["boxes"]
        B, N, _ = pred_logits.shape
        results = []
        for b in range(B):
            probs = pred_logits[b].softmax(-1)
            scores, labels = probs[:, :-1].max(-1)
            keep = scores > score_thr
            boxes = pred_boxes[b, keep]
            h, w = x.shape[2:]
            boxes_xyxy = torch.stack([
                (boxes[:, 0] - boxes[:, 2] / 2) * w,
                (boxes[:, 1] - boxes[:, 3] / 2) * h,
                (boxes[:, 0] + boxes[:, 2] / 2) * w,
                (boxes[:, 1] + boxes[:, 3] / 2) * h,
            ], dim=-1)
            results.append({
                "boxes": boxes_xyxy,
                "scores": scores[keep],
                "labels": labels[keep],
            })
        return results

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "DETR",
            "num_classes": self.num_classes,
            "num_queries": self.num_queries,
            "total_params": total,
            "trainable_params": trainable,
            "params_mb": total * 4 / (1024 ** 2),
        }
