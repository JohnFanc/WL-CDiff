from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(pred - target)))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Global SSIM over 2D arrays (dependency-free fallback)."""
    x = np.asarray(pred, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    data_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))
    if data_range <= 1e-8:
        data_range = 1.0

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ux = x.mean()
    uy = y.mean()
    vx = ((x - ux) ** 2).mean()
    vy = ((y - uy) ** 2).mean()
    cxy = ((x - ux) * (y - uy)).mean()

    num = (2 * ux * uy + c1) * (2 * cxy + c2)
    den = (ux**2 + uy**2 + c1) * (vx + vy + c2)
    return float(num / (den + 1e-12))
