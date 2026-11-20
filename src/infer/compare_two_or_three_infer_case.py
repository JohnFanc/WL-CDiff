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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    fig: plt.Figure,
    ax: plt.Axes,
    arr: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    metrics: dict[str, float] | None,
) -> None:
    im = ax.imshow(arr, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("ln K", fontsize=11)

    if metrics is not None:
        banner = (
            f"RMSE={metrics['rmse']:.4f}    "
            f"MAE={metrics['mae']:.4f}    "
            f"SSIM={metrics['ssim']:.4f}"
        )
        ax.text(
            0.5, -0.12, 
            banner,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#E8F3FF",
                "edgecolor": "#4C78A8",
                "linewidth": 1.2,
            },
        )


def save_comparison(
    true_lnk: np.ndarray,
    pred_maps: list[np.ndarray],
    run_names: list[str],
    metrics_list: list[dict[str, float]],
    out_path: str | Path,
    dpi: int,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    all_maps = [true_lnk] + pred_maps
    vmin = float(min(m.min() for m in all_maps))
    vmax = float(max(m.max() for m in all_maps))
    
    num_preds = len(pred_maps)

    if num_preds == 2:
        # 【修改点 1】: 如果是 2 个结果，采用 1行3列的横向布局，画板设为扁长型
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.35)

        draw_panel(fig, axes[0], true_lnk, "True", vmin, vmax, metrics=None)
        draw_panel(fig, axes[1], pred_maps[0], run_names[0], vmin, vmax, metrics_list[0])
        draw_panel(fig, axes[2], pred_maps[1], run_names[1], vmin, vmax, metrics_list[1])
        
    elif num_preds == 3:
        # 【修改点 2】: 如果是 3 个结果，采用 2行2列的布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.35)

        draw_panel(fig, axes[0, 0], true_lnk, "True", vmin, vmax, metrics=None)
        draw_panel(fig, axes[0, 1], pred_maps[0], run_names[0], vmin, vmax, metrics_list[0])
        draw_panel(fig, axes[1, 0], pred_maps[1], run_names[1], vmin, vmax, metrics_list[1])
        draw_panel(fig, axes[1, 1], pred_maps[2], run_names[2], vmin, vmax, metrics_list[2])
    else:
        raise ValueError(f"不支持的预测结果数量：{num_preds}，请配置 2 或 3 个 infer_runs。")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
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