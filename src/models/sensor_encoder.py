from __future__ import annotations

import math

import torch
import torch.nn as nn


def fourier_encode(coords: torch.Tensor, num_bands: int = 8, max_freq: float = 8.0) -> torch.Tensor:
    """
    coords: (..., 2), normalized into [0, 1]
    returns: (..., 2 + 2*2*num_bands)
    """
    freqs = torch.linspace(1.0, max_freq, num_bands, device=coords.device)
    ang = coords[..., None] * freqs[None, None, :] * (2.0 * math.pi)  # (...,2,B)
    sin = torch.sin(ang)
    cos = torch.cos(ang)
    return torch.cat([coords, sin.flatten(-2), cos.flatten(-2)], dim=-1)


class SensorTemporalEncoder(nn.Module):
    """Per-sensor temporal encoder: (B,S,T,3) -> (B,S,64)."""

    def __init__(
        self,
        in_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_steps: int = 51,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.time_embed = nn.Parameter(torch.randn(max_steps, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs_feat: torch.Tensor) -> torch.Tensor:
        # obs_feat: (B,S,T,3)
        b, s, t, _ = obs_feat.shape
        x = self.input_proj(obs_feat)
        x = x + self.time_embed[:t].view(1, 1, t, -1)
        x = x.view(b * s, t, -1)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return x.view(b, s, -1)


class SensorSetEncoder(nn.Module):
    """
    Set-level encoder with coordinate Fourier features:
      sensor token: (B,S,64) -> (B,S,128)
      global token: (B,256)
    """

    def __init__(
        self,
        sensor_dim: int = 64,
        coord_bands: int = 8,
        model_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        global_dim: int = 256,
        grid_size: int = 50,
    ) -> None:
        super().__init__()
        coord_dim = 2 + 4 * coord_bands
        self.grid_size = float(grid_size - 1)
        self.coord_bands = coord_bands

        self.sensor_proj = nn.Linear(sensor_dim, model_dim)
        self.coord_proj = nn.Linear(coord_dim, model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.token_norm = nn.LayerNorm(model_dim)
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, global_dim),
            nn.SiLU(),
            nn.Linear(global_dim, global_dim),
        )

    def forward(self, sensor_tokens: torch.Tensor, sensor_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sensor_tokens: (B,S,64), sensor_xy: (S,2) or (B,S,2)
        b, s, _ = sensor_tokens.shape
        if sensor_xy.dim() == 2:
            sensor_xy = sensor_xy.unsqueeze(0).expand(b, -1, -1)
        if sensor_xy.max() > 1.0:
            sensor_xy = sensor_xy / self.grid_size

        coord_feat = fourier_encode(sensor_xy, num_bands=self.coord_bands)
        x = self.sensor_proj(sensor_tokens) + self.coord_proj(coord_feat)
        x = self.set_encoder(x)
        x = self.token_norm(x)
        global_token = self.global_mlp(x.mean(dim=1))
        return x, global_token


class SensorConditionEncoder(nn.Module):
    def __init__(self, temporal_cfg: dict, set_cfg: dict) -> None:
        super().__init__()
        self.temporal = SensorTemporalEncoder(**temporal_cfg)
        self.set_encoder = SensorSetEncoder(sensor_dim=temporal_cfg["d_model"], **set_cfg)

    def forward(self, obs_feat: torch.Tensor, sensor_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tkn = self.temporal(obs_feat)
        return self.set_encoder(tkn, sensor_xy)
