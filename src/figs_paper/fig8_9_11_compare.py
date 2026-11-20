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
from src.utils.io import ensure_dir
from src.utils.metrics import mae, rmse, ssim
from src.figs_paper import style

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    "infer_runs": [
        {
            "name": "36 observation wells",
            "dir": "outputs/infer/exp_v1_baseline/v1",
        },
        {
            "name": "18 observation wells",
            "dir": "outputs/infer/exp_v1_s18/v1_s18",
        },
        {
            "name": "4 observation wells",
            "dir": "outputs/infer/exp_v1_s04/v1",
        },
        # 提示：如果你只想对比两个结果，把下面的字典注释或删除即可
        # {
        #     "name": "E5 Ablation",
        #     "dir": "outputs/infer/exp_v1_s36_ablation_E5/v1",
        # },
    ],
    "sample_id": 1506,
    "truth_field_dirs": [
        "dataset/v1/test/fields",
    ],
    "field_kind": "auto",  # auto | lnk | k
    "pred_file_name": "lnK_mean.npy",
    "grid_size": 50,
    "figure_dpi": 200,
    "out_path": "outputs/compare/case_compare/sample_1552_compare.png",
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


def draw_panel(
    ax: plt.Axes,
    arr: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    metrics: dict[str, float] | None,
    panel_tag: str | None = None,
) -> "plt.cm.ScalarMappable":
    im = ax.imshow(arr, cmap=style.VIRIDIS, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=style.pt(14), fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if panel_tag:
        style.panel_label(ax, panel_tag)
    if metrics is not None:
        style.metric_banner(ax, metrics["rmse"], metrics["mae"], metrics["ssim"])
    return im


def save_comparison(
    true_lnk: np.ndarray,
    pred_maps: list[np.ndarray],
    run_names: list[str],
    metrics_list: list[dict[str, float]],
    out_path: str | Path,
    dpi: int,
    sample_id: int | None = None,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    style.apply_style()

    all_maps = [true_lnk] + pred_maps
    vmin, vmax = style.shared_clim(all_maps)
    num_preds = len(pred_maps)
    tags = ["(a)", "(b)", "(c)", "(d)"]
    panels = [(true_lnk, "Reference", None)] + [
        (pred_maps[i], run_names[i], metrics_list[i]) for i in range(num_preds)
    ]

    if num_preds == 2:
        # Reference + 2 estimates -> 1x3 row
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), constrained_layout=True)
        axlist = list(axes)
    elif num_preds == 3:
        # Figure 8: Reference + 3 estimates in one row.  The wide layout keeps
        # the four panels at a consistent size and matches the visual scale of
        # the typical-case panels in Figure 6 more closely than the old 2x2 grid.
        fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.9), constrained_layout=True)
        axlist = list(axes)
    else:
        raise ValueError(f"不支持的预测结果数量：{num_preds}，请配置 2 或 3 个 infer_runs。")

    im = None
    for ax, tag, (arr, title, met) in zip(axlist, tags, panels):
        im = draw_panel(ax, arr, title, vmin, vmax, met, tag)

    # single shared colorbar for the whole figure (common color scale)
    colorbar_fraction = 0.024 if num_preds == 3 else 0.046
    colorbar_shrink = 0.82 if num_preds == 3 else 0.92
    cbar = fig.colorbar(
        im,
        ax=axlist,
        fraction=colorbar_fraction,
        pad=0.015,
        shrink=colorbar_shrink,
    )
    cbar.set_label(style.LNK_LABEL, fontsize=style.pt(12), fontweight="bold")
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    cfg = dict(USER_CFG)
    run_cfgs = list(cfg["infer_runs"])
    
    # 【修改点 3】: 解除严格为 3 的限制，允许 2 或 3 
    if len(run_cfgs) not in (2, 3):
        raise ValueError("USER_CFG['infer_runs'] 必须包含 2 个或 3 个推理结果的配置。")

    sample_id = int(cfg["sample_id"])
    pred_file_name = str(cfg["pred_file_name"])
    grid_size = int(cfg["grid_size"])
    dpi = int(cfg["figure_dpi"])
    out_path = Path(cfg["out_path"])

    truth_by_id = collect_field_by_id(
        [str(x) for x in cfg["truth_field_dirs"]],
        field_kind=str(cfg["field_kind"]),
    )
    if sample_id not in truth_by_id:
        raise KeyError(f"sample_id={sample_id} not found in truth fields.")

    run_names: list[str] = []
    pred_maps: list[np.ndarray] = []
    metrics_list: list[dict[str, float]] = []

    true_lnk = read_lnk_file(truth_by_id[sample_id], grid_size=grid_size).astype(np.float64)

    for run_cfg in run_cfgs:
        run_name = str(run_cfg["name"])
        arrays_dir = resolve_arrays_dir(run_cfg["dir"])
        pred_by_id = collect_pred_by_id(arrays_dir, pred_file_name)
        if sample_id not in pred_by_id:
            raise KeyError(f"sample_id={sample_id} not found under {arrays_dir}")

        pred_lnk = np.load(pred_by_id[sample_id]).astype(np.float64)
        run_names.append(run_name)
        pred_maps.append(pred_lnk)
        metrics_list.append(case_metrics(pred_lnk, true_lnk))

    # 【修改点 4】: 替换为新函数 save_comparison
    save_comparison(
        true_lnk=true_lnk,
        pred_maps=pred_maps,
        run_names=run_names,
        metrics_list=metrics_list,
        out_path=out_path,
        dpi=dpi,
        sample_id=sample_id,
    )

    print(f"[Info] sample_id={sample_id}")
    for run_name, run_cfg, metrics in zip(run_names, run_cfgs, metrics_list):
        print(
            f"[Info] {run_name} <- {resolve_arrays_dir(run_cfg['dir'])} | "
            f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} SSIM={metrics['ssim']:.4f}"
        )
    print(f"[OK] figure -> {out_path}")


if __name__ == "__main__":
    main()
