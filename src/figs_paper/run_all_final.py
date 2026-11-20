from __future__ import annotations

"""
Generate manuscript-ready Figures 4-11 with Times New Roman text.

Run from any directory:
    python -m src.figs_paper.run_all_final

Optional output directory:
    python -m src.figs_paper.run_all_final ^
        --out-dir src/figs_paper/out_paper_tnr
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib import font_manager

from src.figs_paper import final_layouts as F
from src.figs_paper import fig4_7_layout as L
from src.figs_paper import fig5_10_obseries as S
from src.figs_paper import fig8_9_11_compare as C
from src.figs_paper import style
from src.utils.io import ensure_dir


DEFAULT_OUT = PROJECT_ROOT / "src" / "figs_paper" / "out_paper_tnr"


def _resolve_output_dir(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _check_font() -> Path:
    style.apply_style()
    font_path = Path(
        font_manager.findfont("Times New Roman", fallback_to_default=False)
    )
    print(f"[Info] Times New Roman -> {font_path}")
    return font_path


def _render_fig4(out_dir: Path) -> None:
    L.USER_CFG.update(
        sensors_csv="configs/sensors_36.csv",
        out_path=str(out_dir / "fig04_study_area.png"),
    )
    L.main()


def _render_fig5(out_dir: Path) -> None:
    S.USER_CFG.update(
        ob_file="dataset/v1/test/obData/Head1506.npz",
        feature="h_abs",
        sensor_ids=None,
        inset_clean_file=None,
        out_path=str(out_dir / "fig05_obs_series_clean.png"),
    )
    S.main()


def _render_fig8(out_dir: Path) -> None:
    C.USER_CFG.clear()
    C.USER_CFG.update(
        truth_field_dirs=["dataset/v1/test/fields"],
        field_kind="auto",
        pred_file_name="lnK_mean.npy",
        grid_size=50,
        figure_dpi=300,
        infer_runs=[
            {
                "name": "36-well estimate",
                "dir": "outputs/infer/exp_v1_baseline/v1",
            },
            {
                "name": "18-well estimate",
                "dir": "outputs/infer/exp_v1_s18/v1_s18",
            },
            {
                "name": "4-well estimate",
                "dir": "outputs/infer/exp_v1_s04/v1",
            },
        ],
        sample_id=1506,
        out_path=str(out_dir / "fig08_density_compare_id1506.png"),
    )
    C.main()


def _render_fig10(out_dir: Path) -> None:
    S.USER_CFG.update(
        ob_file="dataset/v1/test/obData_s36_noise_s003/Head1506.npz",
        feature="h_abs",
        sensor_ids=None,
        inset_clean_file="dataset/v1/test/obData/Head1506.npz",
        out_path=str(out_dir / "fig10_obs_series_noisy.png"),
    )
    S.main()


def main(out_dir: str | Path = DEFAULT_OUT) -> Path:
    os.chdir(PROJECT_ROOT)
    out_dir = _resolve_output_dir(out_dir)
    ensure_dir(out_dir)
    _check_font()

    _render_fig4(out_dir)
    _render_fig5(out_dir)
    F.render_fig6(out_dir / "fig06_typical_baseline.png")
    F.render_fig7(out_dir / "fig07_layout_18_4_wells.png")
    _render_fig8(out_dir)
    F.render_fig9(out_dir / "fig09_ablation_composite.png")
    _render_fig10(out_dir)
    F.render_fig11(out_dir / "fig11_noise_composite.png")

    expected = [
        "fig04_study_area.png",
        "fig05_obs_series_clean.png",
        "fig06_typical_baseline.png",
        "fig07_layout_18_4_wells.png",
        "fig08_density_compare_id1506.png",
        "fig09_ablation_composite.png",
        "fig10_obs_series_noisy.png",
        "fig11_noise_composite.png",
    ]
    missing = [name for name in expected if not (out_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing generated figures: {missing}")

    print(f"[OK] generated Figures 4-11 -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help="Output directory for Figures 4-11",
    )
    args = parser.parse_args()
    main(args.out_dir)
