from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from pathlib import Path

import matplotlib
import numpy as np

from src.data.sensor_utils import collect_field_by_id, parse_sample_id, read_lnk_file
from src.utils.io import ensure_dir, write_csv
from src.utils.metrics import mae, rmse, ssim

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    "exp_a_name": "Baseline",
    "exp_b_name": "Noise Input",
 
    "exp_a_dir": "outputs/infer/exp_v1_baseline/v1",
    "exp_b_dir": "outputs/infer/exp_v1_baseline/v1_noise_s003",
    "truth_field_dirs": [
        "dataset/v1/test/fields",
    ],
    "field_kind": "auto",  # auto | lnk | k
    "pred_file_name": "lnK_mean.npy",
    "max_samples": None,  # None -> use all overlapping samples
    "out_dir": "outputs/compare/exp_v1_baseline_vs_exp_v1_noise_s003",
    "bins": 20,
    "grid_size": 50,
}


def _collect_pred_by_id(infer_dir: Path, pred_file_name: str) -> dict[int, Path]:
    arr_dir = infer_dir / "arrays"
    if not arr_dir.exists():
        raise FileNotFoundError(f"Missing arrays directory: {arr_dir}")

    out: dict[int, Path] = {}
    for sample_dir in sorted(arr_dir.glob("sample_*")):
        pred_path = sample_dir / pred_file_name
        if not pred_path.exists():
            continue
        sid = parse_sample_id(sample_dir.name)
        out[sid] = pred_path
    return out


def _compute_rows(
    ids: list[int],
    pred_a_by_id: dict[int, Path],
    pred_b_by_id: dict[int, Path],
    truth_by_id: dict[int, str],
    grid_size: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows: list[dict] = []
    err_map_a = np.zeros((grid_size, grid_size), dtype=np.float64)
    err_map_b = np.zeros((grid_size, grid_size), dtype=np.float64)

    for sid in ids:
        pred_a = np.load(pred_a_by_id[sid]).astype(np.float64)
        pred_b = np.load(pred_b_by_id[sid]).astype(np.float64)
        true_lnk = read_lnk_file(truth_by_id[sid], grid_size=grid_size).astype(np.float64)

        pred_a_k = np.exp(pred_a)
        pred_b_k = np.exp(pred_b)
        true_k = np.exp(true_lnk)

        row = {
            "sample_id": sid,
            "rmse_lnk_a": rmse(pred_a, true_lnk),
            "rmse_lnk_b": rmse(pred_b, true_lnk),
            "mae_lnk_a": mae(pred_a, true_lnk),
            "mae_lnk_b": mae(pred_b, true_lnk),
            "ssim_lnk_a": ssim(pred_a, true_lnk),
            "ssim_lnk_b": ssim(pred_b, true_lnk),
            "rmse_k_a": rmse(pred_a_k, true_k),
            "rmse_k_b": rmse(pred_b_k, true_k),
            "mae_k_a": mae(pred_a_k, true_k),
            "mae_k_b": mae(pred_b_k, true_k),
            "ssim_k_a": ssim(pred_a_k, true_k),
            "ssim_k_b": ssim(pred_b_k, true_k),
        }
        row["delta_rmse_lnk_b_minus_a"] = row["rmse_lnk_b"] - row["rmse_lnk_a"]
        row["delta_mae_lnk_b_minus_a"] = row["mae_lnk_b"] - row["mae_lnk_a"]
        row["delta_ssim_lnk_b_minus_a"] = row["ssim_lnk_b"] - row["ssim_lnk_a"]
        row["delta_rmse_k_b_minus_a"] = row["rmse_k_b"] - row["rmse_k_a"]
        row["delta_mae_k_b_minus_a"] = row["mae_k_b"] - row["mae_k_a"]
        row["delta_ssim_k_b_minus_a"] = row["ssim_k_b"] - row["ssim_k_a"]
        rows.append(row)

        err_map_a += np.abs(pred_a - true_lnk)
        err_map_b += np.abs(pred_b - true_lnk)

    n = max(1, len(rows))
    return rows, err_map_a / n, err_map_b / n


def _plot_ssim_distribution(rows: list[dict], exp_a_name: str, exp_b_name: str, out_path: Path, bins: int) -> None:
    a = np.asarray([r["ssim_lnk_a"] for r in rows], dtype=np.float64)
    b = np.asarray([r["ssim_lnk_b"] for r in rows], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    axes[0].hist(a, bins=bins, alpha=0.6, label=exp_a_name, color="#1f77b4")
    axes[0].hist(b, bins=bins, alpha=0.6, label=exp_b_name, color="#d62728")
    axes[0].set_title("SSIM(lnK) Distribution")
    axes[0].set_xlabel("SSIM")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    axes[0].legend()

    # Matplotlib>=3.9 uses tick_labels; old versions use labels.
    try:
        axes[1].boxplot([a, b], tick_labels=[exp_a_name, exp_b_name], showmeans=True)
    except TypeError:
        axes[1].boxplot([a, b], labels=[exp_a_name, exp_b_name], showmeans=True)
    axes[1].set_title("SSIM(lnK) Boxplot")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_paired_scatter(rows: list[dict], exp_a_name: str, exp_b_name: str, out_path: Path) -> None:
    metric_pairs = [
        ("rmse_lnk_a", "rmse_lnk_b", "RMSE(lnK)", False),
        ("mae_lnk_a", "mae_lnk_b", "MAE(lnK)", False),
        ("ssim_lnk_a", "ssim_lnk_b", "SSIM(lnK)", True),
        ("rmse_k_a", "rmse_k_b", "RMSE(K)", False),
        ("mae_k_a", "mae_k_b", "MAE(K)", False),
        ("ssim_k_a", "ssim_k_b", "SSIM(K)", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    for ax, (ka, kb, title, higher_better) in zip(axes.flatten(), metric_pairs):
        x = np.asarray([r[ka] for r in rows], dtype=np.float64)
        y = np.asarray([r[kb] for r in rows], dtype=np.float64)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        pad = (hi - lo) * 0.05 + 1e-12
        lo -= pad
        hi += pad

        ax.scatter(x, y, s=18, alpha=0.65, c="#2ca02c", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel(exp_a_name)
        ax.set_ylabel(exp_b_name)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        if higher_better:
            win_rate_b = float((y > x).mean())
        else:
            win_rate_b = float((y < x).mean())
        ax.text(
            0.02,
            0.98,
            f"{exp_b_name} better: {win_rate_b * 100:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_mean_abs_error_map(
    mae_map_a: np.ndarray, mae_map_b: np.ndarray, exp_a_name: str, exp_b_name: str, out_path: Path
) -> None:
    diff = mae_map_b - mae_map_a
    vmax = max(float(mae_map_a.max()), float(mae_map_b.max()))
    vabs = max(abs(float(diff.min())), abs(float(diff.max())))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    im0 = axes[0].imshow(mae_map_a, cmap="viridis", origin="upper", vmin=0.0, vmax=vmax)
    axes[0].set_title(f"Mean |Pred-True| lnK\n{exp_a_name}")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mae_map_b, cmap="viridis", origin="upper", vmin=0.0, vmax=vmax)
    axes[1].set_title(f"Mean |Pred-True| lnK\n{exp_b_name}")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, cmap="coolwarm", origin="upper", vmin=-vabs, vmax=vabs)
    axes[2].set_title(f"Difference ({exp_b_name} - {exp_a_name})")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _append_summary(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    keys = list(rows[0].keys())
    summary = {"sample_id": -1}
    for k in keys:
        if k == "sample_id":
            continue
        summary[k] = float(np.mean([r[k] for r in rows]))
    return rows + [summary]


def main() -> None:
    cfg = dict(USER_CFG)
    exp_a_name = str(cfg["exp_a_name"])
    exp_b_name = str(cfg["exp_b_name"])
    exp_a_dir = Path(cfg["exp_a_dir"])
    exp_b_dir = Path(cfg["exp_b_dir"])
    out_dir = ensure_dir(cfg["out_dir"])
    bins = int(cfg["bins"])
    grid_size = int(cfg["grid_size"])
    pred_file_name = str(cfg["pred_file_name"])
    max_samples = cfg.get("max_samples")

    pred_a_by_id = _collect_pred_by_id(exp_a_dir, pred_file_name)
    pred_b_by_id = _collect_pred_by_id(exp_b_dir, pred_file_name)
    truth_by_id = collect_field_by_id(list(cfg["truth_field_dirs"]), field_kind=str(cfg["field_kind"]))

    common_ids = sorted(set(pred_a_by_id) & set(pred_b_by_id) & set(truth_by_id))
    if max_samples is not None:
        common_ids = common_ids[: int(max_samples)]

    if not common_ids:
        raise ValueError("No overlapping sample ids found between the two result dirs and truth field dirs.")

    print(f"[Info] {exp_a_name} samples: {len(pred_a_by_id)}")
    print(f"[Info] {exp_b_name} samples: {len(pred_b_by_id)}")
    print(f"[Info] truth samples: {len(truth_by_id)}")
    print(f"[Info] overlapping samples used: {len(common_ids)}")

    rows, mae_map_a, mae_map_b = _compute_rows(
        ids=common_ids,
        pred_a_by_id=pred_a_by_id,
        pred_b_by_id=pred_b_by_id,
        truth_by_id=truth_by_id,
        grid_size=grid_size,
    )

    rows_with_summary = _append_summary(rows)
    csv_path = Path(out_dir) / "compare_metrics.csv"
    write_csv(csv_path, fieldnames=rows_with_summary[0].keys(), rows=rows_with_summary)

    _plot_ssim_distribution(
        rows=rows,
        exp_a_name=exp_a_name,
        exp_b_name=exp_b_name,
        out_path=Path(out_dir) / "ssim_distribution_lnk.png",
        bins=bins,
    )
    _plot_paired_scatter(
        rows=rows,
        exp_a_name=exp_a_name,
        exp_b_name=exp_b_name,
        out_path=Path(out_dir) / "paired_scatter_metrics.png",
    )
    _plot_mean_abs_error_map(
        mae_map_a=mae_map_a,
        mae_map_b=mae_map_b,
        exp_a_name=exp_a_name,
        exp_b_name=exp_b_name,
        out_path=Path(out_dir) / "mean_abs_error_map_lnk.png",
    )

    mean_row = rows_with_summary[-1]
    print(f"[OK] metrics csv -> {csv_path}")
    print(f"[OK] figures -> {out_dir}")
    print(
        "[Summary] "
        f"SSIM(lnK) {exp_a_name}={mean_row['ssim_lnk_a']:.4f}, "
        f"{exp_b_name}={mean_row['ssim_lnk_b']:.4f}, "
        f"delta(b-a)={mean_row['delta_ssim_lnk_b_minus_a']:.4f}"
    )


if __name__ == "__main__":
    main()
