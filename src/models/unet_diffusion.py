from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / max(half - 1, 1)))
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaGNResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)

        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond1 = nn.Linear(cond_dim, in_ch * 2)
        self.cond2 = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    @staticmethod
    def _mod(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale[..., None, None]) + shift[..., None, None]

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, b1 = self.cond1(cond).chunk(2, dim=-1)
        h = self._mod(self.norm1(x), s1, b1)
        h = F.silu(h)
        h = self.conv1(h)

        s2, b2 = self.cond2(cond).chunk(2, dim=-1)
        h = self._mod(self.norm2(h), s2, b2)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class MidCrossAttention(nn.Module):
    def __init__(self, feat_ch: int, sensor_dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feat_ch)
        self.kv_proj = nn.Linear(sensor_dim, feat_ch)
        self.attn = nn.MultiheadAttention(feat_ch, heads, batch_first=True)

    def forward(self, x: torch.Tensor, sensor_tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)  # (B,HW,C)
        qn = self.norm(q)
        kv = self.kv_proj(sensor_tokens)
        out, _ = self.attn(qn, kv, kv)
        q = q + out
        return q.transpose(1, 2).reshape(b, c, h, w)


class ConditionalUNet(nn.Module):
    """
    v-pred UNet:
      input  x_t: (B,1,50,50), cond_map: (B,96,50,50)
      output v:   (B,1,50,50)
    """

    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 96,
        channels: tuple[int, int, int] | list[int] = (64, 128, 256),
        time_dim: int = 256,
        global_dim: int = 256,
        sensor_dim: int = 128,
        attn_heads: int = 4,
        use_global_token: bool = True,
        use_mid_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.use_global_token = bool(use_global_token)
        self.use_mid_cross_attn = bool(use_mid_cross_attn)

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.global_proj = nn.Linear(global_dim, time_dim)

        self.in_conv = nn.Conv2d(in_channels + cond_channels, c1, 3, padding=1)
        self.down1 = AdaGNResBlock(c1, c1, time_dim)
        self.ds1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.down2 = AdaGNResBlock(c2, c2, time_dim)
        self.ds2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)

        self.mid1 = AdaGNResBlock(c3, c3, time_dim)
        self.mid_attn = MidCrossAttention(c3, sensor_dim=sensor_dim, heads=attn_heads)
        self.mid2 = AdaGNResBlock(c3, c3, time_dim)

        self.up_proj1 = nn.Conv2d(c3, c2, 3, padding=1)
        self.up1 = AdaGNResBlock(c2 + c2, c2, time_dim)
        self.up_proj2 = nn.Conv2d(c2, c1, 3, padding=1)
        self.up2 = AdaGNResBlock(c1 + c1, c1, time_dim)

        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_map: Optional[torch.Tensor],
        sensor_tokens: torch.Tensor,
        global_token: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.time_mlp(self.time_embed(t))
        if self.use_global_token:
            cond = cond + self.global_proj(global_token)

        if cond_map is None:
            x = x_t
        else:
            x = torch.cat([x_t, cond_map], dim=1)
        h0 = self.in_conv(x)
        h1 = self.down1(h0, cond)  # (B,c1,50,50)

        h2_in = self.ds1(h1)
        h2 = self.down2(h2_in, cond)  # (B,c2,25,25)

        h3_in = self.ds2(h2)  # (B,c3,13,13)
        m = self.mid1(h3_in, cond)
        if self.use_mid_cross_attn:
            m = self.mid_attn(m, sensor_tokens)
        m = self.mid2(m, cond)

        u1 = F.interpolate(m, size=h2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up_proj1(u1)
        u1 = self.up1(torch.cat([u1, h2], dim=1), cond)

        u2 = F.interpolate(u1, size=h1.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up_proj2(u2)
        u2 = self.up2(torch.cat([u2, h1], dim=1), cond)

        out = self.out_conv(F.silu(self.out_norm(u2)))
        return out
