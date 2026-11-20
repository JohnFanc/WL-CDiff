from __future__ import annotations

"""
Final paper layouts for Figures 6, 7, 9, and 11.

These functions reproduce the composite arrangements used in the current
manuscript:
  - Fig. 6: two rows (Reference / Estimated) by three samples.
  - Fig. 7: two well-layout panels without the extra shared legend.
  - Fig. 9: three samples by three columns
    (Reference / WL-CDiff estimate / Unconditional output).
  - Fig. 11: three samples by three columns
    (Reference / Noiseless estimate / Noisy estimate).

All text styling is inherited from src.figs_paper.style.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.sensor_utils import collect_field_by_id, read_lnk_file
from src.figs_paper import style
from src.figs_paper.fig4_7_layout import _draw_panel
from src.figs_paper.fig8_9_11_compare import (
    case_metrics,
    collect_pred_by_id,
    resolve_arrays_dir,
)
from src.utils.io import ensure_dir


def _truth_map(
    sample_id: int,
    truth_field_dirs: list[str] | None = None,
    grid_size: int = 50,
) -> np.ndarray:
    truth_field_dirs = truth_field_dirs or ["dataset/v1/test/fields"]
    truth_by_id = collect_field_by_id(truth_field_dirs, field_kind="auto")
    if sample_id not in truth_by_id:
        raise KeyError(f"sample_id={sample_id} not found in truth fields")
    return read_lnk_file(truth_by_id[sample_id], grid_size=grid_size).astype(np.float64)


def _prediction_map(
    infer_dir: str,
    sample_id: int,
    pred_file_name: str = "lnK_mean.npy",
) -> np.ndarray:
    arrays_dir = resolve_arrays_dir(infer_dir)
    pred_by_id = collect_pred_by_id(arrays_dir, pred_file_name)
    if sample_id not in pred_by_id:
        raise KeyError(f"sample_id={sample_id} not found under {arrays_dir}")
    return np.load(pred_by_id[sample_id]).astype(np.float64)


def _clean_field_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def render_fig6(
    out_path: str | Path,
    sample_ids: tuple[int, int, int] = (1506, 1507, 1508),
    infer_dir: str = "outputs/infer/exp_v1_baseline/v1",
) -> None:
    """Render the manuscript's 2 x 3 typical-case layout."""
    style.apply_style()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig = plt.figure(figsize=(13.8, 7.75))
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[1.0, 0.045, 1.0, 0.045, 1.0, 0.045],
        left=0.055,
        right=0.985,
        top=0.985,
        bottom=0.135,
        wspace=0.13,
        hspace=0.12,
    )

    tags = ("(a)", "(b)", "(c)")
    for col, sample_id in enumerate(sample_ids):
        true_lnk = _truth_map(sample_id)
        pred_lnk = _prediction_map(infer_dir, sample_id)
        metrics = case_metrics(pred_lnk, true_lnk)
        vmin, vmax = style.shared_clim([true_lnk, pred_lnk])

        ax_true = fig.add_subplot(grid[0, 2 * col])
        ax_pred = fig.add_subplot(grid[1, 2 * col])
        cax_true = fig.add_subplot(grid[0, 2 * col + 1])
        cax_pred = fig.add_subplot(grid[1, 2 * col + 1])

        im_true = ax_true.imshow(
            true_lnk, cmap=style.VIRIDIS, origin="upper", vmin=vmin, vmax=vmax
        )
        im_pred = ax_pred.imshow(
            pred_lnk, cmap=style.VIRIDIS, origin="upper", vmin=vmin, vmax=vmax
        )
        _clean_field_axis(ax_true)
        _clean_field_axis(ax_pred)

        if col == 0:
            ax_true.set_ylabel("Reference", fontsize=style.pt(14))
            ax_pred.set_ylabel("Estimated", fontsize=style.pt(14))

        for image, cax in ((im_true, cax_true), (im_pred, cax_pred)):
            cbar = fig.colorbar(image, cax=cax)
            cbar.ax.set_title(style.LNK_LABEL, fontsize=style.pt(10), pad=5)

        style.metric_banner(
            ax_pred,
            metrics["rmse"],
            metrics["mae"],
            metrics["ssim"],
            y=-0.035,
        )
        ax_pred.text(
            0.5,
            -0.135,
            tags[col],
            transform=ax_pred.transAxes,
            ha="center",
            va="top",
            fontsize=style.pt(13),
            fontweight="bold",
        )
        print(
            f"[Info] Fig.6 {tags[col]} sample {sample_id}: "
            f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} "
            f"SSIM={metrics['ssim']:.4f}"
        )

    fig.savefig(out_path, dpi=style.DPI)
    plt.close(fig)
    print(f"[OK] final Fig.6 -> {out_path}")


def render_fig7(
    out_path: str | Path,
    sensor_csvs: tuple[str, str] = (
        "configs/sensors_18.csv",
        "configs/sensors_04.csv",
    ),
) -> None:
    """Render the manuscript's two-panel well layout without an extra legend."""
    style.apply_style()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    titles = ("(a) 18 monitoring wells", "(b) 4 monitoring wells")
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.35), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.86, bottom=0.13, wspace=0.30)

    for ax, csv_path, title in zip(axes, sensor_csvs, titles):
        _draw_panel(
            ax,
            csv_path,
            show_ylabel=True,
            show_left_noflow=True,
            show_right_noflow=True,
        )
        # Keep the panel title above the H=105 m boundary label drawn at y=1.04.
        ax.set_title(title, fontsize=style.pt(13), fontweight="bold", y=1.12, pad=0)

    fig.savefig(out_path, dpi=style.DPI)
    plt.close(fig)
    print(f"[OK] final Fig.7 -> {out_path}")


def _render_three_by_three(
    out_path: str | Path,
    sample_ids: tuple[int, int, int],
    run_specs: tuple[tuple[str, str], tuple[str, str]],
) -> None:
    style.apply_style()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig = plt.figure(figsize=(12.6, 12.8))
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        left=0.04,
        right=0.965,
        top=0.96,
        bottom=0.035,
        wspace=0.16,
        hspace=0.22,
    )
    tags = ("(a)", "(b)", "(c)")

    for row, sample_id in enumerate(sample_ids):
        true_lnk = _truth_map(sample_id)
        pred_maps = [
            _prediction_map(infer_dir, sample_id)
            for _, infer_dir in run_specs
        ]
        metrics_list = [case_metrics(pred, true_lnk) for pred in pred_maps]
        vmin, vmax = style.shared_clim([true_lnk, *pred_maps])
        maps = [true_lnk, *pred_maps]
        titles = ["Reference", run_specs[0][0], run_specs[1][0]]

        images = []
        for col, (field, title) in enumerate(zip(maps, titles)):
            ax = fig.add_subplot(grid[row, col])
            image = ax.imshow(
                field,
                cmap=style.VIRIDIS,
                origin="upper",
                vmin=vmin,
                vmax=vmax,
            )
            images.append(image)
            _clean_field_axis(ax)
            if row == 0:
                ax.set_title(title, fontsize=style.pt(14), fontweight="bold", pad=9)
                style.panel_label(ax, tags[col], dx=-0.055, dy=1.035)
            if col > 0:
                metrics = metrics_list[col - 1]
                style.metric_banner(
                    ax,
                    metrics["rmse"],
                    metrics["mae"],
                    metrics["ssim"],
                    y=-0.045,
                )

        cax = fig.add_subplot(grid[row, 3])
        cbar = fig.colorbar(images[-1], cax=cax)
        cbar.set_label(style.LNK_LABEL, fontsize=style.pt(11), fontweight="bold")
        for tick_label in cbar.ax.get_yticklabels():
            tick_label.set_fontweight("bold")

        print(
            f"[Info] sample {sample_id}: "
            + " | ".join(
                f"{name}: RMSE={met['rmse']:.4f} "
                f"MAE={met['mae']:.4f} SSIM={met['ssim']:.4f}"
                for (name, _), met in zip(run_specs, metrics_list)
            )
        )

    fig.savefig(out_path, dpi=style.DPI)
    plt.close(fig)
    print(f"[OK] final composite -> {out_path}")


def render_fig9(out_path: str | Path) -> None:
    """Render the manuscript's 3 x 3 full-vs-unconditional composite."""
    _render_three_by_three(
        out_path,
        sample_ids=(1501, 1505, 1507),
        run_specs=(
            ("WL-CDiff estimate", "outputs/infer/ablation/full"),
            ("Unconditional output", "outputs/infer/ablation/E5_uncond"),
        ),
    )


def render_fig11(out_path: str | Path) -> None:
    """Render the manuscript's 3 x 3 noiseless-vs-noisy composite."""
    _render_three_by_three(
        out_path,
        sample_ids=(1507, 1553, 1555),
        run_specs=(
            ("Noiseless estimate", "outputs/infer/exp_v1_baseline/v1"),
            ("Noisy estimate", "outputs/infer/exp_v1_baseline/v1_noise_s003"),
        ),
    )
