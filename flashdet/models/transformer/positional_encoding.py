"""Fixed 2-D sinusoidal positional encoding for spatial feature maps."""

import torch
import torch.nn as nn


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
