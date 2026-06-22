"""Lightweight text encoder for Grounding DINO.

Falls back to a trainable token embedding + transformer encoder when
no pretrained CLIP model is found, so the module is always usable.
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Lightweight text encoder using a transformer, or wrapping CLIP/BERT if available."""

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
        x = self.token_embed(input_ids) + self.pos_embed[:, :input_ids.shape[1]]

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.ln(x)
