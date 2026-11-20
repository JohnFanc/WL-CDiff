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
    # Direct path to one inference arrays directory.
    # Example: outputs/infer/exp_v1_baseline/v1/arrays
    "infer_arrays_dir": "outputs/infer/exp_v1_s18/v1_s18/arrays",
    # Ground-truth lnK/K field directory.
    "truth_field_dirs": [
        "dataset/v1/test/fields",
    ],
    "field_kind": "auto",  # auto | lnk | k
    "pred_file_name": "lnK_mean.npy",
    "grid_size": 50,
    "max_samples": None,  # None -> use all overlapping samples
    # None -> <infer_arrays_dir>/../evaluation
    "out_dir": None,
    "table_png_name": "metrics_summary_table.png",
    "detail_csv_name": "metrics_detail.csv",
    "summary_csv_name": "metrics_summary.csv",
    "title": "Inference Evaluation Summary",
}


METRIC_DISPLAY_NAMES = {
    "rmse_lnk": "RMSE (lnK)",
    "mae_lnk": "MAE (lnK)",
    "ssim_lnk": "SSIM (lnK)",
    "rmse_k": "RMSE (K)",
    "mae_k": "MAE (K)",
    "ssim_k": "SSIM (K)",
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
    raise FileNotFoundError(f"Cannot find sample_* directories or arrays subdir under: {path}")


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
        raise ValueError(f"No prediction files named {pred_file_name} found in {arrays_dir}")
    return out


def compute_rows(
    ids: list[int],
    pred_by_id: dict[int, Path],
    truth_by_id: dict[int, str],
    grid_size: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for sid in ids:
        pred_lnk = np.load(pred_by_id[sid]).astype(np.float64)
        true_lnk = read_lnk_file(truth_by_id[sid], grid_size=grid_size).astype(np.float64)

        pred_k = np.exp(pred_lnk)
        true_k = np.exp(true_lnk)

        rows.append(
            {
                "sample_id": sid,
                "rmse_lnk": rmse(pred_lnk, true_lnk),
                "mae_lnk": mae(pred_lnk, true_lnk),
                "ssim_lnk": ssim(pred_lnk, true_lnk),
                "rmse_k": rmse(pred_k, true_k),
                "mae_k": mae(pred_k, true_k),
                "ssim_k": ssim(pred_k, true_k),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, float | int]]) -> list[dict[str, float | str]]:
    metric_keys = ["rmse_lnk", "mae_lnk", "ssim_lnk", "rmse_k", "mae_k", "ssim_k"]
    summary_rows: list[dict[str, float | str]] = []
    for key in metric_keys:
        values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        summary_rows.append(
            {
                "metric": str(METRIC_DISPLAY_NAMES.get(key, key)),
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
        )
    return summary_rows


def save_summary_table(
    summary_rows: list[dict[str, float | str]],
    out_path: str | Path,
    title: str,
    num_samples: int,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    col_labels = ["Metric", "Mean", "Std"]
    cell_text = []
    for row in summary_rows:
        cell_text.append(
            [
                str(row["metric"]),
                f"{float(row['mean']):.6f}",
                f"{float(row['std']):.6f}",
            ]
        )

    fig_h = max(3.6, 1.0 + 0.55 * len(cell_text))
    fig, ax = plt.subplots(1, 1, figsize=(8.6, fig_h), constrained_layout=True)
    ax.axis("off")
    ax.set_title(f"{title}\nSamples: {num_samples}", fontsize=12, pad=10)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DCEBFA")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8FBFF" if r % 2 == 1 else "#EEF5FC")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = dict(USER_CFG)
    arrays_dir = resolve_arrays_dir(cfg["infer_arrays_dir"])
    truth_dirs = [str(x) for x in cfg["truth_field_dirs"]]
    pred_file_name = str(cfg["pred_file_name"])
    grid_size = int(cfg["grid_size"])
    max_samples = cfg.get("max_samples")

    if cfg.get("out_dir") is None:
        out_dir = arrays_dir.parent / "evaluation"
    else:
        out_dir = Path(cfg["out_dir"])
    ensure_dir(out_dir)

    pred_by_id = collect_pred_by_id(arrays_dir, pred_file_name)
    truth_by_id = collect_field_by_id(truth_dirs, field_kind=str(cfg["field_kind"]))

    common_ids = sorted(set(pred_by_id) & set(truth_by_id))
    if max_samples is not None:
        common_ids = common_ids[: int(max_samples)]
    if not common_ids:
        raise ValueError("No overlapping sample ids found between inference arrays and truth fields.")

    rows = compute_rows(common_ids, pred_by_id, truth_by_id, grid_size)
    summary_rows = summarize_rows(rows)

    detail_csv = out_dir / str(cfg["detail_csv_name"])
    summary_csv = out_dir / str(cfg["summary_csv_name"])
    table_png = out_dir / str(cfg["table_png_name"])

    write_csv(detail_csv, fieldnames=rows[0].keys(), rows=rows)
    write_csv(summary_csv, fieldnames=summary_rows[0].keys(), rows=summary_rows)
    save_summary_table(
        summary_rows=summary_rows,
        out_path=table_png,
        title=str(cfg["title"]),
        num_samples=len(rows),
    )

    print(f"[Info] arrays_dir={arrays_dir}")
    print(f"[Info] truth_dirs={truth_dirs}")
    print(f"[Info] overlapping samples={len(rows)}")
    print(f"[OK] detail csv -> {detail_csv}")
    print(f"[OK] summary csv -> {summary_csv}")
    print(f"[OK] summary table -> {table_png}")


if __name__ == "__main__":
    main()
