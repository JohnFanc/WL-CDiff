from __future__ import annotations

import torch
import torch.nn as nn

from src.models.sensor_encoder import fourier_encode


class GridConditionalDecoder(nn.Module):
    """
    2500 grid queries cross-attend sensor tokens -> cond_map(B,96,50,50).
    """

    def __init__(
        self,
        grid_size: int = 50,
        sensor_dim: int = 128,
        cond_dim: int = 96,
        num_heads: int = 4,
        coord_bands: int = 8,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.coord_bands = coord_bands
        self.sensor_dim = sensor_dim
        self.cond_dim = cond_dim

        yy, xx = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )
        grid_xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2) / float(grid_size - 1)
        self.register_buffer("grid_xy", grid_xy, persistent=False)

        coord_dim = 2 + 4 * coord_bands
        self.query_proj = nn.Sequential(
            nn.Linear(coord_dim, sensor_dim),
            nn.SiLU(),
            nn.Linear(sensor_dim, sensor_dim),
        )
        self.cross_attn = nn.MultiheadAttention(sensor_dim, num_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(sensor_dim),
            nn.Linear(sensor_dim, sensor_dim),
            nn.SiLU(),
            nn.Linear(sensor_dim, cond_dim),
        )

    def forward(self, sensor_tokens: torch.Tensor) -> torch.Tensor:
        # sensor_tokens: (B,S,128)
        b = sensor_tokens.shape[0]
        q_feat = fourier_encode(self.grid_xy[None, :, :].expand(b, -1, -1), self.coord_bands)
        q = self.query_proj(q_feat)  # (B,2500,128)
        h, _ = self.cross_attn(q, sensor_tokens, sensor_tokens)
        h = self.out_proj(h)  # (B,2500,96)
        cond = h.transpose(1, 2).reshape(b, self.cond_dim, self.grid_size, self.grid_size)
        return cond
