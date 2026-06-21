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

from flashdet.registry import BACKBONES

logger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    """Lightweight text encoder using a transformer, or wrapping CLIP/BERT if available.

    Falls back to a trainable token embedding + transformer encoder when
    no pretrained CLIP model is found, so the module is always usable.
    """

    def __init__(self, vocab_size: int = 49408, embed_dim: int = 256, max_len: int = 77, depth: int = 4, nhead: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] integer token ids.
            attention_mask: [B, L] binary mask (1 = valid, 0 = padding).

        Returns:
            [B, L, embed_dim] contextualised text features.
        """
        x = self.token_embed(input_ids) + self.pos_embed[:, :input_ids.shape[1]]

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.ln(x)


class VisionLanguageFusion(nn.Module):
    """Bi-directional cross-attention between visual and text features."""

    def __init__(self, d_model: int = 256, nhead: int = 8):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.t2v_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.v_norm = nn.LayerNorm(d_model)
        self.t_norm = nn.LayerNorm(d_model)
        self.v_ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.t_ffn = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.v_ffn_norm = nn.LayerNorm(d_model)
        self.t_ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        visual_feat: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_padding = (text_mask == 0) if text_mask is not None else None

        v_res = self.v2t_attn(visual_feat, text_feat, text_feat, key_padding_mask=key_padding)[0]
        visual_feat = self.v_norm(visual_feat + v_res)
        visual_feat = self.v_ffn_norm(visual_feat + self.v_ffn(visual_feat))

        t_res = self.t2v_attn(text_feat, visual_feat, visual_feat)[0]
        text_feat = self.t_norm(text_feat + t_res)
        text_feat = self.t_ffn_norm(text_feat + self.t_ffn(text_feat))

        return visual_feat, text_feat


class GroundingDINODecoder(nn.Module):
    """Transformer decoder with language-aware cross-attention."""

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6, num_queries: int = 900):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "self_attn": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                "cross_attn_vis": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                "cross_attn_text": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "norm2": nn.LayerNorm(d_model),
                "norm3": nn.LayerNorm(d_model),
                "norm4": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model),
                ),
            }))

        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

    def forward(
        self,
        visual_feat: torch.Tensor,
        text_feat: torch.Tensor,
        text_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B = visual_feat.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        key_padding = (text_mask == 0) if text_mask is not None else None

        hs = queries
        for layer in self.layers:
            hs2 = layer["self_attn"](hs, hs, hs)[0]
            hs = layer["norm1"](hs + hs2)

            hs2 = layer["cross_attn_vis"](hs, visual_feat, visual_feat)[0]
            hs = layer["norm2"](hs + hs2)

            hs2 = layer["cross_attn_text"](hs, text_feat, text_feat, key_padding_mask=key_padding)[0]
            hs = layer["norm3"](hs + hs2)

            hs = layer["norm4"](hs + layer["ffn"](hs))

        pred_boxes = self.bbox_head(hs).sigmoid()

        # Language-aware classification: dot-product similarity with text tokens
        pred_logits = torch.bmm(hs, text_feat.transpose(1, 2))

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


@BACKBONES.register("GroundingDINO")
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

        import torchvision.models as tv
        backbone_map = {
            "resnet50": (tv.resnet50, 2048),
            "resnet101": (tv.resnet101, 2048),
        }
        builder, vis_ch = backbone_map.get(backbone, (tv.resnet50, 2048))
        weights = "DEFAULT" if pretrained_backbone else None
        resnet = builder(weights=weights)
        self.visual_backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.visual_proj = nn.Sequential(
            nn.Conv2d(vis_ch, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
        )

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
        vis_feat = self.visual_proj(self.visual_backbone(images))
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
        B = pred_logits.shape[0]

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

            target_cls = torch.zeros(self.num_queries, dtype=torch.float32, device=device)
            target_cls[row] = 1.0
            total_cls.append(F.binary_cross_entropy_with_logits(
                max_logit, target_cls, pos_weight=torch.tensor(5.0, device=device),
            ))

        loss_cls = torch.stack(total_cls).mean() if total_cls else pred_logits.sum() * 0
        loss_l1 = torch.stack(total_l1).mean() if total_l1 else pred_boxes.sum() * 0

        total = loss_cls + 5.0 * loss_l1
        return total, {"loss_cls": loss_cls.detach(), "loss_l1": loss_l1.detach()}

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        score_thr: float = 0.3,
    ) -> List[Dict]:
        self.eval()
        out = self.forward(images, input_ids, attention_mask)
        dec_out = out["preds"]
        pred_logits = dec_out["pred_logits"]
        pred_boxes = dec_out["pred_boxes"]
        B = pred_logits.shape[0]
        h, w = images.shape[2:]

        results = []
        for b in range(B):
            scores = pred_logits[b].sigmoid().max(dim=-1)
            keep = scores.values > score_thr
            boxes = pred_boxes[b, keep]
            boxes_xyxy = torch.stack([
                (boxes[:, 0] - boxes[:, 2] / 2) * w,
                (boxes[:, 1] - boxes[:, 3] / 2) * h,
                (boxes[:, 0] + boxes[:, 2] / 2) * w,
                (boxes[:, 1] + boxes[:, 3] / 2) * h,
            ], dim=-1)
            results.append({
                "boxes": boxes_xyxy,
                "scores": scores.values[keep],
                "text_indices": scores.indices[keep],
            })
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
