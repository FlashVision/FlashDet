"""RT-DETR Transformer decoder with IoU-aware query selection."""

from typing import Dict, List

import torch
import torch.nn as nn


class RTDETRDecoder(nn.Module):
    """RT-DETR Transformer decoder with IoU-aware query selection."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        num_queries: int = 300,
        num_classes: int = 80,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.num_classes = num_classes

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.enc_cls_head = nn.Linear(d_model, num_classes)
        self.enc_bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        self.enc_iou_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

        self.dec_cls_head = nn.Linear(d_model, num_classes)
        self.dec_bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )

        self.query_pos_head = nn.Sequential(
            nn.Linear(4, d_model * 2), nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, encoder_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat_list = []
        for feat in encoder_features:
            feat_list.append(feat.flatten(2).permute(0, 2, 1))
        memory = torch.cat(feat_list, dim=1)
        B, N, C = memory.shape

        enc_cls = self.enc_cls_head(memory)
        enc_bbox = self.enc_bbox_head(memory).sigmoid()
        enc_iou = self.enc_iou_head(memory).squeeze(-1).sigmoid()

        cls_scores = enc_cls.sigmoid().max(dim=-1).values
        selection_scores = cls_scores * enc_iou
        _, topk_idx = selection_scores.topk(self.num_queries, dim=1)

        query_embed = torch.gather(memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
        ref_boxes = torch.gather(enc_bbox, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

        query_pos = self.query_pos_head(ref_boxes)

        hs = self.decoder(query_embed + query_pos, memory)
        hs = self.decoder_norm(hs)

        dec_cls = self.dec_cls_head(hs)
        dec_bbox = (self.dec_bbox_head(hs) + self._inverse_sigmoid(ref_boxes)).sigmoid()

        return {
            "pred_logits": dec_cls,
            "pred_boxes": dec_bbox,
            "enc_logits": enc_cls,
            "enc_boxes": enc_bbox,
            "topk_idx": topk_idx,
        }

    @staticmethod
    def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        x = x.clamp(min=eps, max=1 - eps)
        return torch.log(x / (1 - x))
