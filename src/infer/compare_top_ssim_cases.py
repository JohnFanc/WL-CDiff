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
    # Either an infer root dir (contains arrays/) or the arrays dir itself.
    "infer_dir_a": "outputs/infer/exp_v1_baseline/v1",
    "infer_dir_b": "outputs/infer/exp_v1_s36_ablation_E2/v1",
    "name_a": "Baseline",
    "name_b": "E2 Ablation",
    "truth_field_dirs": [
        "dataset/v1/test/fields",
    ],
    "field_kind": "auto",  # auto | lnk | k
    "pred_file_name": "lnK_mean.npy",
    "grid_size": 50,
    "top_k": 10,
    "out_dir": "outputs/compare/top_ssim_gap_cases_baseline_vs_e2",
    "figure_dpi": 200,
}


def resolve_arrays_dir(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Inference result directory not found: {path}")
    if any(p.is_dir() and p.name.startswith("sample_") for p in path.iterdir()):
        return path
    arrays_dir = path / "arrays"
    if arrays_dir.exists():
        return arrays_dir
    raise FileNotFoundError(f"Cannot find arrays directory under: {path}")


def collect_pred_by_id(arrays_dir: Path, pred_file_name: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for sample_dir in sorted(arrays_dir.glob("sample_*")):
        if not sample_dir.is_dir():
            continue
        pred_path = sample_dir / pred_file_name
        if not pred_path.exists():
            continue
        sid = parse_sample_id(sample_dir.name)
        out[sid] = pred_path
    if not out:
        raise ValueError(f"No {pred_file_name} found in {arrays_dir}")
    return out


def case_metrics(pred_lnk: np.ndarray, true_lnk: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(pred_lnk, true_lnk),
        "mae": mae(pred_lnk, true_lnk),
        "ssim": ssim(pred_lnk, true_lnk),
    }


def build_rows(
    pred_a_by_id: dict[int, Path],
    pred_b_by_id: dict[int, Path],
    truth_by_id: dict[int, str],
    grid_size: int,
) -> list[dict[str, float | int]]:
    common_ids = sorted(set(pred_a_by_id) & set(pred_b_by_id) & set(truth_by_id))
    rows: list[dict[str, float | int]] = []
    for sid in common_ids:
        pred_a = np.load(pred_a_by_id[sid]).astype(np.float64)
        pred_b = np.load(pred_b_by_id[sid]).astype(np.float64)
        true_lnk = read_lnk_file(truth_by_id[sid], grid_size=grid_size).astype(np.float64)

        m_a = case_metrics(pred_a, true_lnk)
        m_b = case_metrics(pred_b, true_lnk)
        delta_signed = m_a["ssim"] - m_b["ssim"]
        rows.append(
            {
                "sample_id": sid,
                "ssim_a": m_a["ssim"],
                "ssim_b": m_b["ssim"],
                "rmse_a": m_a["rmse"],
                "rmse_b": m_b["rmse"],
                "mae_a": m_a["mae"],
                "mae_b": m_b["mae"],
                "delta_ssim_signed_a_minus_b": delta_signed,
                "delta_ssim_abs": abs(delta_signed),
            }
        )
    return rows


def _panel_with_colorbar(fig: plt.Figure, ax: plt.Axes, arr: np.ndarray, title: str, cmap: str, vmin: float, vmax: float) -> None:
    im = ax.imshow(arr, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ln K")


def add_metric_banner(ax: plt.Axes, rmse_value: float, mae_value: float, ssim_value: float) -> None:
    banner = f"RMSE={rmse_value:.4f}   MAE={mae_value:.4f}   SSIM={ssim_value:.4f}"
    ax.text(
        0.5,
        -0.08,
        banner,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#E8F3FF",
            "edgecolor": "#4C78A8",
            "linewidth": 1.0,
        },
    )


def save_triplet_case(
    true_lnk: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    name_a: str,
    name_b: str,
    out_path: str | Path,
    sample_id: int,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    dpi: int,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    vmin = float(min(true_lnk.min(), pred_a.min(), pred_b.min()))
    vmax = float(max(true_lnk.max(), pred_a.max(), pred_b.max()))

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.9), constrained_layout=True)
    _panel_with_colorbar(fig, axes[0], true_lnk, "True", "viridis", vmin, vmax)
    _panel_with_colorbar(fig, axes[1], pred_a, name_a, "viridis", vmin, vmax)
    _panel_with_colorbar(fig, axes[2], pred_b, name_b, "viridis", vmin, vmax)

    add_metric_banner(axes[1], metrics_a["rmse"], metrics_a["mae"], metrics_a["ssim"])
    add_metric_banner(axes[2], metrics_b["rmse"], metrics_b["mae"], metrics_b["ssim"])

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    cfg = dict(USER_CFG)
    arrays_dir_a = resolve_arrays_dir(cfg["infer_dir_a"])
    arrays_dir_b = resolve_arrays_dir(cfg["infer_dir_b"])
    truth_dirs = [str(x) for x in cfg["truth_field_dirs"]]
    out_dir = ensure_dir(cfg["out_dir"])
    pred_file_name = str(cfg["pred_file_name"])
    grid_size = int(cfg["grid_size"])
    top_k = int(cfg["top_k"])
    dpi = int(cfg["figure_dpi"])
    name_a = str(cfg["name_a"])
    name_b = str(cfg["name_b"])

    pred_a_by_id = collect_pred_by_id(arrays_dir_a, pred_file_name)
    pred_b_by_id = collect_pred_by_id(arrays_dir_b, pred_file_name)
    truth_by_id = collect_field_by_id(truth_dirs, field_kind=str(cfg["field_kind"]))

    rows = build_rows(pred_a_by_id, pred_b_by_id, truth_by_id, grid_size)
    if not rows:
        raise ValueError("No overlapping sample ids found.")

    rows_sorted = sorted(rows, key=lambda x: float(x["delta_ssim_abs"]), reverse=True)
    top_rows = rows_sorted[:top_k]

    csv_path = Path(out_dir) / "top_ssim_gap_ids.csv"
    write_csv(csv_path, fieldnames=top_rows[0].keys(), rows=top_rows)

    fig_dir = ensure_dir(Path(out_dir) / "figures")
    for row in top_rows:
        sid = int(row["sample_id"])
        true_lnk = read_lnk_file(truth_by_id[sid], grid_size=grid_size).astype(np.float64)
        pred_a = np.load(pred_a_by_id[sid]).astype(np.float64)
        pred_b = np.load(pred_b_by_id[sid]).astype(np.float64)
        metrics_a = case_metrics(pred_a, true_lnk)
        metrics_b = case_metrics(pred_b, true_lnk)

        save_triplet_case(
            true_lnk=true_lnk,
            pred_a=pred_a,
            pred_b=pred_b,
            name_a=name_a,
            name_b=name_b,
            out_path=fig_dir / f"sample_{sid:04d}_triplet_compare.png",
            sample_id=sid,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            dpi=dpi,
        )

    print(f"[Info] arrays_dir_a={arrays_dir_a}")
    print(f"[Info] arrays_dir_b={arrays_dir_b}")
    print(f"[Info] overlapping samples={len(rows)}")
    print(f"[OK] top-{len(top_rows)} ids csv -> {csv_path}")
    print("[Top IDs]")
    for row in top_rows:
        print(
            f"ID={int(row['sample_id'])} "
            f"SSIM_A={float(row['ssim_a']):.4f} "
            f"SSIM_B={float(row['ssim_b']):.4f} "
            f"|delta|={float(row['delta_ssim_abs']):.4f}"
        )
    print(f"[OK] figures -> {fig_dir}")


if __name__ == "__main__":
    main()
