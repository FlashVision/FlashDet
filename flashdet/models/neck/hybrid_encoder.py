"""RT-DETR Hybrid Encoder: AIFI (intra-scale) + CCFM (cross-scale fusion)."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.models.layers import CSPRepLayer


class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction.

    Applies self-attention within a single scale feature map using a standard
    transformer encoder layer with 2-D positional encoding.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _build_2d_sincos_pos(self, H: int, W: int, d_model: int, device: torch.device) -> torch.Tensor:
        half = d_model // 2
        dim = torch.arange(half // 2, dtype=torch.float32, device=device)
        dim = 10000.0 ** (2 * (dim // 2) / (half // 2))

        pos_y = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1) / dim
        pos_x = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1) / dim

        pe_y = torch.cat([pos_y.sin(), pos_y.cos()], dim=-1).unsqueeze(1).expand(-1, W, -1)
        pe_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1).unsqueeze(0).expand(H, -1, -1)

        return torch.cat([pe_y, pe_x], dim=-1).reshape(H * W, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pos = self._build_2d_sincos_pos(H, W, C, x.device)

        tokens = x.flatten(2).permute(0, 2, 1)
        tokens = tokens + pos.unsqueeze(0)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


class HybridEncoder(nn.Module):
    """RT-DETR Hybrid Encoder: AIFI (intra-scale) + CCFM (cross-scale fusion).

    Takes multi-scale features from the backbone, applies AIFI to the
    highest-resolution feature, then fuses features top-down and
    bottom-up through CCFM layers.
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_dim: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_encoder_layers: int = 1,
        num_csp_blocks: int = 3,
        expansion: float = 1.0,
    ):
        super().__init__()
        self.num_scales = len(in_channels)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for ch in in_channels
        ])

        self.aifi = AIFI(hidden_dim, nhead, dim_feedforward, num_encoder_layers)

        self.top_down_layers = nn.ModuleList()
        self.top_down_csp = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        self.bottom_up_csp = nn.ModuleList()

        for _ in range(self.num_scales - 1):
            self.top_down_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ))
            self.top_down_csp.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_csp_blocks, expansion))

        for _ in range(self.num_scales - 1):
            self.bottom_up_layers.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ))
            self.bottom_up_csp.append(CSPRepLayer(hidden_dim * 2, hidden_dim, num_csp_blocks, expansion))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        projected = [proj(f) for proj, f in zip(self.input_proj, features)]

        projected[-1] = self.aifi(projected[-1])

        td_outs = [None] * self.num_scales
        td_outs[-1] = projected[-1]
        for i in range(self.num_scales - 2, -1, -1):
            up = F.interpolate(self.top_down_layers[i](td_outs[i + 1]),
                               size=projected[i].shape[2:], mode="nearest")
            td_outs[i] = self.top_down_csp[i](torch.cat([up, projected[i]], dim=1))

        bu_outs = [None] * self.num_scales
        bu_outs[0] = td_outs[0]
        for i in range(self.num_scales - 1):
            down = self.bottom_up_layers[i](bu_outs[i])
            bu_outs[i + 1] = self.bottom_up_csp[i](torch.cat([down, td_outs[i + 1]], dim=1))

        return bu_outs
