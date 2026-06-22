"""Bi-directional cross-attention between visual and text features."""

from typing import Tuple

import torch
import torch.nn as nn


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
