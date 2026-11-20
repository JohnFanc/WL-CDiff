from __future__ import annotations

import torch
import torch.nn.functional as F


def fft_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(pred.squeeze(1), norm="ortho")
    target_fft = torch.fft.rfft2(target.squeeze(1), norm="ortho")
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


def compute_total_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    x0_pred: torch.Tensor,
    x0_true: torch.Tensor,
    weights: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    lv = F.mse_loss(v_pred, v_target)
    lx0 = F.l1_loss(x0_pred, x0_true)
    lfft = fft_l1_loss(x0_pred, x0_true)
    ltv = tv_loss(x0_pred)

    total = (
        float(weights["lv"]) * lv
        + float(weights["lx0"]) * lx0
        + float(weights["lfft"]) * lfft
        + float(weights["ltv"]) * ltv
    )
    return total, {"lv": lv, "lx0": lx0, "lfft": lfft, "ltv": ltv}
