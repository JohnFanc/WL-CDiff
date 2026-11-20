from __future__ import annotations

"""
Shared figure style for the paper (experiment-section figures 4-11).

Centralizes journal-grade defaults so every regenerated figure is visually
consistent: one font family / size set, 300 dpi, unified colormap and lnK label,
panel (a)(b)(c) tags, shared color scale helpers, and a metric banner style.

This module is new (not a copy); the per-figure scripts in this folder are copies
of the original plotting scripts with only style + labeling adjustments applied.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# --- global constants -------------------------------------------------------
DPI = 300
VIRIDIS = "viridis"
LNK_LABEL = "lnK"             # unified with the manuscript (lnK, not "ln K")
CELL_M = 10.0                 # grid cell size (m); 50x50 grid -> 500 m x 500 m
GRID_SIZE = 50
FONT_SCALE = 1.30             # enlarged for legibility after manuscript scaling

METRIC_BBOX = {
    "boxstyle": "round,pad=0.35",
    "facecolor": "#EEF4FB",
    "edgecolor": "#3F6FA8",
    "linewidth": 1.0,
}


def pt(size: float) -> float:
    """Scale a reference font size for the manuscript-ready raster figures."""
    return size * FONT_SCALE


def apply_style() -> None:
    """Apply a single consistent rcParams set for all paper figures."""
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": pt(12),
        "axes.titlesize": pt(14),
        "axes.labelsize": pt(12),
        "xtick.labelsize": pt(10.5),
        "ytick.labelsize": pt(10.5),
        "legend.fontsize": pt(10.5),
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.3,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax: plt.Axes, text: str, dx: float = -0.02, dy: float = 1.04) -> None:
    """Bold (a)/(b)/(c) tag at the top-left, just outside the axes."""
    ax.text(
        dx, dy, text,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=pt(14), fontweight="bold",
    )


def metric_banner(ax: plt.Axes, rmse_value: float, mae_value: float, ssim_value: float,
                  y: float = -0.055) -> None:
    """Rounded metric banner centered below an image panel."""
    banner = f"RMSE={rmse_value:.4f}  MAE={mae_value:.4f}  SSIM={ssim_value:.4f}"
    ax.text(
        0.5, y, banner,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=pt(10.5), fontweight="bold",
        bbox=METRIC_BBOX,
    )


def style_field_axes_meters(ax: plt.Axes, grid_size: int = GRID_SIZE,
                            cell_m: float = CELL_M, show_ticks: bool = True) -> None:
    """Label a 50x50 field axis in metres (0..500 m) instead of bare grid indices."""
    if not show_ticks:
        ax.set_xticks([]); ax.set_yticks([])
        return
    ticks_idx = np.linspace(0, grid_size, 6)          # 0,10,...,50
    ticks_m = (ticks_idx * cell_m).astype(int)        # 0,100,...,500
    ax.set_xticks(ticks_idx - 0.5)
    ax.set_yticks(ticks_idx - 0.5)
    ax.set_xticklabels(ticks_m)
    ax.set_yticklabels(ticks_m)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def shared_clim(maps: list[np.ndarray]) -> tuple[float, float]:
    """Common (vmin, vmax) so reference and estimated panels share one scale."""
    vmin = float(min(m.min() for m in maps))
    vmax = float(max(m.max() for m in maps))
    return vmin, vmax
