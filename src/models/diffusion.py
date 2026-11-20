from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from src.models.cond_decoder import GridConditionalDecoder
from src.models.sensor_encoder import SensorConditionEncoder
from src.models.unet_diffusion import ConditionalUNet


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    out = a.gather(0, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        sensor_encoder: nn.Module,
        cond_decoder: Optional[nn.Module],
        unet: nn.Module,
        *,
        grid_size: int = 50,
        global_dim: int = 256,
        global_cond_channels: int = 0,
    ) -> None:
        super().__init__()
        self.sensor_encoder = sensor_encoder
        self.cond_decoder = cond_decoder
        self.unet = unet
        self.grid_size = int(grid_size)
        self.global_cond_channels = int(global_cond_channels)

        if self.cond_decoder is None and self.global_cond_channels > 0:
            self.global_to_map = nn.Sequential(
                nn.LayerNorm(global_dim),
                nn.Linear(global_dim, self.global_cond_channels),
            )
        else:
            self.global_to_map = None

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, obs_feat: torch.Tensor, sensor_xy: torch.Tensor) -> torch.Tensor:
        sensor_tokens, global_token = self.sensor_encoder(obs_feat, sensor_xy)
        if self.cond_decoder is not None:
            cond_map = self.cond_decoder(sensor_tokens)
        elif self.global_to_map is not None:
            g = self.global_to_map(global_token)  # (B,Cg)
            cond_map = g[:, :, None, None].expand(-1, -1, self.grid_size, self.grid_size)
        else:
            # xt-only variant: no cond_map branch, only global_token -> AdaGN conditioning.
            cond_map = None
        return self.unet(x_t, t, cond_map, sensor_tokens, global_token)


class DiffusionScheduler(nn.Module):
    def __init__(self, timesteps: int = 300) -> None:
        super().__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        a = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        s = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return a * x0 + s * noise

    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        s = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return a * noise - s * x0

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        s = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return a * x_t - s * v

    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        a = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        s = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return s * x_t + a * v

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        obs_feat: torch.Tensor,
        sensor_xy: torch.Tensor,
        grid_size: int = 50,
        num_steps: int = 100,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        device = obs_feat.device
        b = obs_feat.shape[0]
        x = torch.randn((b, 1, grid_size, grid_size), device=device)
        step_indices = torch.linspace(self.timesteps - 1, 0, num_steps, device=device).long()

        for i, t_cur in enumerate(step_indices):
            t_batch = torch.full((b,), int(t_cur.item()), device=device, dtype=torch.long)
            v_cond = model(x, t_batch, obs_feat, sensor_xy)
            if cfg_scale != 1.0:
                zeros = torch.zeros_like(obs_feat)
                v_uncond = model(x, t_batch, zeros, sensor_xy)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond

            x0 = self.predict_x0_from_v(x, t_batch, v)
            if clip_x0:
                x0 = x0.clamp(-4.0, 4.0)

            if i == len(step_indices) - 1:
                x = x0
                break

            t_next = step_indices[i + 1]
            a_t = self.alphas_cumprod[int(t_cur.item())]
            a_next = self.alphas_cumprod[int(t_next.item())]
            eps = self.predict_eps_from_v(x, t_batch, v)

            sigma = eta * torch.sqrt((1 - a_next) / (1 - a_t) * (1 - a_t / a_next))
            noise = torch.randn_like(x) if eta > 0 else 0.0
            c = torch.sqrt(torch.clamp(1 - a_next - sigma**2, min=0.0))
            x = torch.sqrt(a_next) * x0 + c * eps + sigma * noise
        return x


def build_model_from_config(cfg: dict) -> ConditionalDiffusionModel:
    model_cfg = copy.deepcopy(cfg["model"])
    ablation_cfg = copy.deepcopy(model_cfg.get("ablation", {}))
    disable_token_to_cond_map = bool(ablation_cfg.get("disable_token_to_cond_map", False))
    global_broadcast_channels = int(ablation_cfg.get("global_broadcast_channels", 8))
    disable_global_to_adagn = bool(ablation_cfg.get("disable_global_to_adagn", False))
    disable_sensor_mid_cross_attn = bool(ablation_cfg.get("disable_sensor_mid_cross_attn", False))

    temporal_cfg = copy.deepcopy(model_cfg["sensor_temporal"])
    set_cfg = copy.deepcopy(model_cfg["sensor_set"])
    sensor_encoder = SensorConditionEncoder(
        temporal_cfg=temporal_cfg,
        set_cfg=set_cfg,
    )

    unet_cfg = copy.deepcopy(model_cfg["unet"])
    cond_decoder = None
    global_cond_channels = 0

    if disable_token_to_cond_map:
        # Ablation mode: remove token->cond_map decoder branch.
        # If global_broadcast_channels == 0, UNet input keeps xt-only 1 channel.
        if global_broadcast_channels < 0:
            raise ValueError("model.ablation.global_broadcast_channels must be >= 0")
        unet_cfg["cond_channels"] = int(global_broadcast_channels)
        global_cond_channels = int(global_broadcast_channels)
    else:
        cond_cfg = copy.deepcopy(model_cfg["cond_decoder"])
        cond_decoder = GridConditionalDecoder(**cond_cfg)
        # Keep UNet input channels consistent with cond_decoder output channels.
        unet_cfg["cond_channels"] = int(cond_cfg["cond_dim"])

    unet_cfg["use_global_token"] = not disable_global_to_adagn
    unet_cfg["use_mid_cross_attn"] = not disable_sensor_mid_cross_attn
    unet = ConditionalUNet(**unet_cfg)

    return ConditionalDiffusionModel(
        sensor_encoder=sensor_encoder,
        cond_decoder=cond_decoder,
        unet=unet,
        grid_size=int(cfg["data"].get("grid_size", 50)),
        global_dim=int(set_cfg.get("global_dim", unet_cfg.get("global_dim", 256))),
        global_cond_channels=global_cond_channels,
    )
