from __future__ import annotations

"""
Figure 6 (Section 4.1): typical reference vs estimated fields.

The original caption did not record the sample IDs; they were recovered by matching
the SSIM/RMSE printed in the published figure against the baseline metrics_log.csv:
    row (a) = sample 1506 (RMSE 0.2644, SSIM 0.8772)
    row (b) = sample 1507 (RMSE 0.2858, SSIM 0.9279)
    row (c) = sample 1508 (RMSE 0.3306, SSIM 0.8487)

This script is new (the original only had a per-sample comparison helper in
src.utils.viz that auto-scaled the two fields independently); here both fields
share one color scale per row, with (a)(b)(c) row tags and one colorbar per row.

Run from project root the project root:
  python -m src.figs_paper.fig6_typical
"""

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from pathlib import Path

import matplotlib
import numpy as np

from src.data.sensor_utils import collect_field_by_id, read_lnk_file
from src.utils.io import ensure_dir
from src.utils.metrics import mae, rmse, ssim
from src.figs_paper import style

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    "infer_dir": "outputs/infer/exp_v1_baseline/v1",
    "sample_ids": [1506, 1507, 1508],
    "truth_field_dirs": ["dataset/v1/test/fields"],
    "field_kind": "auto",
    "pred_file_name": "lnK_mean.npy",
    "grid_size": 50,
    "out_path": "src/figs_paper/out/fig06_typical_baseline.png",
    "figure_dpi": 300,
}


def _arrays_dir(path_str: str | Path) -> Path:
    path = Path(path_str)
    if (path / "arrays").exists():
        return path / "arrays"
    return path


def main() -> None:
    style.apply_style()
    cfg = dict(USER_CFG)
    arrays_dir = _arrays_dir(cfg["infer_dir"])
    ids = [int(s) for s in cfg["sample_ids"]]
    grid_size = int(cfg["grid_size"])
    pred_name = str(cfg["pred_file_name"])
    out_path = Path(cfg["out_path"])
    ensure_dir(out_path.parent)

    truth_by_id = collect_field_by_id(
        [str(x) for x in cfg["truth_field_dirs"]], field_kind=str(cfg["field_kind"])
    )

    n = len(ids)
    tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    fig, axes = plt.subplots(n, 2, figsize=(8.0, 3.9 * n), constrained_layout=True)
    if n == 1:
        axes = axes[None, :]

    for r, sid in enumerate(ids):
        if sid not in truth_by_id:
            raise KeyError(f"sample_id={sid} not in truth fields")
        true_lnk = read_lnk_file(truth_by_id[sid], grid_size=grid_size).astype(np.float64)
        pred_path = arrays_dir / f"sample_{sid}" / pred_name
        if not pred_path.exists():
            raise FileNotFoundError(f"missing prediction: {pred_path}")
        pred_lnk = np.load(pred_path).astype(np.float64)

        vmin, vmax = style.shared_clim([true_lnk, pred_lnk])
        ax_l, ax_r = axes[r, 0], axes[r, 1]
        im = ax_l.imshow(true_lnk, cmap=style.VIRIDIS, origin="upper", vmin=vmin, vmax=vmax)
        ax_r.imshow(pred_lnk, cmap=style.VIRIDIS, origin="upper", vmin=vmin, vmax=vmax)
        for ax in (ax_l, ax_r):
            ax.set_xticks([]); ax.set_yticks([])

        if r == 0:
            ax_l.set_title("Reference", fontsize=style.pt(14), pad=8)
            ax_r.set_title("Estimated", fontsize=style.pt(14), pad=8)
        style.panel_label(ax_l, tags[r])

        m = {
            "rmse": rmse(pred_lnk, true_lnk),
            "mae": mae(pred_lnk, true_lnk),
            "ssim": ssim(pred_lnk, true_lnk),
        }
        style.metric_banner(ax_r, m["rmse"], m["mae"], m["ssim"])

        # one shared colorbar per row (each sample has its own value range)
        cbar = fig.colorbar(im, ax=[ax_l, ax_r], fraction=0.046, pad=0.02)
        cbar.set_label(style.LNK_LABEL, fontsize=style.pt(11))

        print(f"[Info] row {tags[r]} sample {sid}: "
              f"RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} SSIM={m['ssim']:.4f}")

    fig.savefig(out_path, dpi=int(cfg["figure_dpi"]))
    plt.close(fig)
    print(f"[OK] figure -> {out_path}")


if __name__ == "__main__":
    main()
