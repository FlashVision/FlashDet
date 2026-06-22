"""Standard Transformer encoder-decoder for DETR."""

import torch
import torch.nn as nn


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
