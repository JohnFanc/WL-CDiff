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
    "infer_runs": [
        {
            "name": "Baseline",
            "dir": "outputs/infer/exp_v1_baseline/v1",
        },
        {
            "name": "s18",
            "dir": "outputs/infer/exp_v1_s18/v1_s18",
        },
        {
            "name": "s04",
            "dir": "outputs/infer/exp_v1_s04/v1",
        }
    ],
    "truth_field_dirs": [
        "dataset/v1/test/fields",
    ],
    "field_kind": "auto",  # auto | lnk | k
    "pred_file_name": "lnK_mean.npy",
    "grid_size": 50,
    "max_samples": None,  # None -> use all overlapping samples
    "out_dir": "outputs/compare/multi_infer_results",
    "title": "Inference Result Comparison",
    "summary_csv_name": "comparison_summary.csv",
    "detail_csv_name": "comparison_detail.csv",
    "table_png_name": "comparison_table.png",
}


METRIC_SPECS = [
    ("rmse_lnk", "RMSE (lnK)", False),
    ("mae_lnk", "MAE (lnK)", False),
    ("ssim_lnk", "SSIM (lnK)", True),
    ("rmse_k", "RMSE (K)", False),
    ("mae_k", "MAE (K)", False),
    ("ssim_k", "SSIM (K)", True),
]


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
        raise ValueError(f"No prediction files named {pred_file_name} found in {arrays_dir}")
    return out


def compute_metrics(pred_lnk: np.ndarray, true_lnk: np.ndarray) -> dict[str, float]:
    pred_k = np.exp(pred_lnk)
    true_k = np.exp(true_lnk)
    return {
        "rmse_lnk": rmse(pred_lnk, true_lnk),
        "mae_lnk": mae(pred_lnk, true_lnk),
        "ssim_lnk": ssim(pred_lnk, true_lnk),
        "rmse_k": rmse(pred_k, true_k),
        "mae_k": mae(pred_k, true_k),
        "ssim_k": ssim(pred_k, true_k),
    }


def build_detail_rows(
    run_name: str,
    pred_by_id: dict[int, Path],
    truth_by_id: dict[int, str],
    sample_ids: list[int],
    grid_size: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for sid in sample_ids:
        pred_lnk = np.load(pred_by_id[sid]).astype(np.float64)
        true_lnk = read_lnk_file(truth_by_id[sid], grid_size=grid_size).astype(np.float64)
        metrics = compute_metrics(pred_lnk, true_lnk)
        row: dict[str, float | int | str] = {
            "run_name": run_name,
            "sample_id": sid,
        }
        row.update(metrics)
        rows.append(row)
    return rows


def summarize_run(run_name: str, detail_rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "run_name": run_name,
        "num_samples": len(detail_rows),
    }
    for metric_key, _, _ in METRIC_SPECS:
        values = np.asarray([float(row[metric_key]) for row in detail_rows], dtype=np.float64)
        summary[f"{metric_key}_mean"] = float(values.mean())
        summary[f"{metric_key}_std"] = float(values.std())
    return summary


def _best_row_indices(summary_rows: list[dict[str, float | int | str]]) -> dict[tuple[int, int], bool]:
    best_cells: dict[tuple[int, int], bool] = {}
    for metric_idx, (metric_key, _, higher_better) in enumerate(METRIC_SPECS, start=1):
        values = np.asarray([float(row[f"{metric_key}_mean"]) for row in summary_rows], dtype=np.float64)
        target = float(values.max() if higher_better else values.min())
        for col_idx, value in enumerate(values, start=1):
            if abs(float(value) - target) <= 1e-12:
                best_cells[(metric_idx, col_idx)] = True
    return best_cells


def build_summary_csv_rows(
    summary_rows: list[dict[str, float | int | str]],
) -> tuple[list[str], list[dict[str, float | str]]]:
    fieldnames = ["Metric"]
    for row in summary_rows:
        run_name = str(row["run_name"])
        fieldnames.extend([f"{run_name} mean", f"{run_name} std"])

    rows: list[dict[str, float | str]] = []
    for metric_key, metric_label, _ in METRIC_SPECS:
        out_row: dict[str, float | str] = {"Metric": metric_label}
        for row in summary_rows:
            run_name = str(row["run_name"])
            out_row[f"{run_name} mean"] = float(row[f"{metric_key}_mean"])
            out_row[f"{run_name} std"] = float(row[f"{metric_key}_std"])
        rows.append(out_row)
    return fieldnames, rows


def save_comparison_table(
    summary_rows: list[dict[str, float | int | str]],
    out_path: str | Path,
    title: str,
    num_common_samples: int,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    run_names = [str(row["run_name"]) for row in summary_rows]
    col_labels = ["Metric"] + run_names
    cell_text = []
    for metric_key, metric_label, _ in METRIC_SPECS:
        values = [metric_label]
        for row in summary_rows:
            mean_v = float(row[f"{metric_key}_mean"])
            std_v = float(row[f"{metric_key}_std"])
            values.append(f"{mean_v:.4f}`n± {std_v:.4f}")
        cell_text.append(values)

    fig_h = max(3.8, 1.4 + 0.58 * len(cell_text))
    fig_w = max(8.8, 2.2 + 2.1 * len(col_labels))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)
    ax.axis("off")
    ax.set_title(f"{title}`nCommon Samples: {num_common_samples}", fontsize=12, pad=10)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.0, 1.8)

    best_cells = _best_row_indices(summary_rows)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#DCEBFA")
            cell.set_text_props(weight="bold")
            continue
        if c == 0:
            cell.set_facecolor("#EEF5FC" if r % 2 == 1 else "#F8FBFF")
            cell.set_text_props(weight="bold")
            continue
        if (r, c) in best_cells:
            cell.set_facecolor("#DDF5E3")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F8FBFF" if r % 2 == 1 else "#EEF5FC")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = dict(USER_CFG)
    infer_runs = list(cfg["infer_runs"])
    if len(infer_runs) < 2:
        raise ValueError("USER_CFG['infer_runs'] must contain at least two inference directories.")

    truth_dirs = [str(x) for x in cfg["truth_field_dirs"]]
    pred_file_name = str(cfg["pred_file_name"])
    grid_size = int(cfg["grid_size"])
    max_samples = cfg.get("max_samples")
    out_dir = ensure_dir(cfg["out_dir"])

    truth_by_id = collect_field_by_id(truth_dirs, field_kind=str(cfg["field_kind"]))
    run_entries: list[dict] = []
    common_ids: set[int] | None = None

    for run_cfg in infer_runs:
        run_name = str(run_cfg["name"])
        arrays_dir = resolve_arrays_dir(run_cfg["dir"])
        pred_by_id = collect_pred_by_id(arrays_dir, pred_file_name)
        ids = set(pred_by_id) & set(truth_by_id)
        common_ids = ids if common_ids is None else (common_ids & ids)
        run_entries.append(
            {
                "run_name": run_name,
                "arrays_dir": arrays_dir,
                "pred_by_id": pred_by_id,
            }
        )

    sample_ids = sorted(common_ids or [])
    if max_samples is not None:
        sample_ids = sample_ids[: int(max_samples)]
    if not sample_ids:
        raise ValueError("No overlapping sample ids found across all inference directories and truth fields.")

    detail_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for run_entry in run_entries:
        run_detail = build_detail_rows(
            run_name=str(run_entry["run_name"]),
            pred_by_id=dict(run_entry["pred_by_id"]),
            truth_by_id=truth_by_id,
            sample_ids=sample_ids,
            grid_size=grid_size,
        )
        detail_rows.extend(run_detail)
        summary_rows.append(summarize_run(str(run_entry["run_name"]), run_detail))

    detail_csv = Path(out_dir) / str(cfg["detail_csv_name"])
    summary_csv = Path(out_dir) / str(cfg["summary_csv_name"])
    table_png = Path(out_dir) / str(cfg["table_png_name"])

    write_csv(detail_csv, fieldnames=detail_rows[0].keys(), rows=detail_rows)
    write_csv(summary_csv, fieldnames=summary_rows[0].keys(), rows=summary_rows)
    save_comparison_table(
        summary_rows=summary_rows,
        out_path=table_png,
        title=str(cfg["title"]),
        num_common_samples=len(sample_ids),
    )

    print(f"[Info] compared runs={len(run_entries)}")
    print(f"[Info] common samples={len(sample_ids)}")
    for run_entry in run_entries:
        print(f"[Info] {run_entry['run_name']} -> {run_entry['arrays_dir']}")
    print(f"[OK] detail csv -> {detail_csv}")
    print(f"[OK] summary csv -> {summary_csv}")
    print(f"[OK] comparison table -> {table_png}")


if __name__ == "__main__":
    main()

