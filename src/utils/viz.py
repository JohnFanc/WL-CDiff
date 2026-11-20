from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from src.utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_triplet(
    true_map: np.ndarray,
    pred_map: np.ndarray,
    diff_map: np.ndarray,
    out_path: str | Path,
    title: str,
    value_cmap: str = "viridis",
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    maps = [true_map, pred_map, diff_map]
    names = ["True", "Pred", "Diff"]
    cmaps = [value_cmap, value_cmap, "coolwarm"]

    for ax, m, n, cm in zip(axes, maps, names, cmaps):
        im = ax.imshow(m, cmap=cm)
        ax.set_title(n)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_lnk_pair_with_metrics(
    true_map: np.ndarray,
    pred_map: np.ndarray,
    out_path: str | Path,
    rmse_value: float,
    mae_value: float,
    ssim_value: float,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), constrained_layout=True)
    maps = [true_map, pred_map]
    names = ["True", "Pred"]
    cmaps = ["viridis", "viridis"]
    cbar_labels = ["ln K", "ln K"]

    for ax, m, n, cm, cbar_label in zip(axes, maps, names, cmaps, cbar_labels):
        im = ax.imshow(m, cmap=cm)
        ax.set_title(n, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    banner = (
        f"RMSE={rmse_value:.4f}   "
        f"MAE={mae_value:.4f}   "
        f"SSIM={ssim_value:.4f}"
    )
    fig.text(
        0.5,
        0.065,
        banner,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#E8F3FF",
            "edgecolor": "#4C78A8",
            "linewidth": 1.2,
        },
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_uncertainty_map(std_map: np.ndarray, out_path: str | Path, title: str = "lnK Std") -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.2), constrained_layout=True)
    im = ax.imshow(std_map, cmap="magma")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
