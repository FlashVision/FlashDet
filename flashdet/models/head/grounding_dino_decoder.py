"""Grounding DINO Transformer decoder with language-aware cross-attention."""

from typing import Dict

import torch
import torch.nn as nn


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
        pred_logits = torch.bmm(hs, text_feat.transpose(1, 2))

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
