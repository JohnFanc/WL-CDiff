from __future__ import annotations

"""
Regenerate the experiment-section figures (4-11) for the paper, using the SAME
test-sample IDs as the published figures, into src/figs_paper/out/.

Run from project root the project root:
  python -m src.figs_paper.run_all
(any env with matplotlib + numpy works; no GPU needed.)

Sample IDs (from the captions of the original 全文.docx):
  fig8  density compare : 1506          (36 / 18 / 4 wells)
  fig9  ablation        : 1501 1505 1507 (full vs unconditional)
  fig11 noise compare   : 1507 1553 1555 (clean vs noisy)
  fig6  typical (S4.1)  : reverse-found by SSIM (no ID in caption)
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

OUT = "src/figs_paper/out"


def run_layout() -> None:
    from src.figs_paper import fig4_7_layout as L
    # 图四: study area + 36 monitoring wells
    L.USER_CFG.update(sensors_csv="configs/sensors_36.csv",
                      out_path=f"{OUT}/fig04_study_area.png")
    L.main()
    # 图七: 18-well and 4-well layouts merged into one figure with a single shared legend
    L.render_combined(
        ["configs/sensors_18.csv", "configs/sensors_04.csv"],
        ["(a) 18 monitoring wells", "(b) 4 monitoring wells"],
        f"{OUT}/fig07_layout_18_4_wells.png",
    )


def run_series() -> None:
    from src.figs_paper import fig5_10_obseries as S
    # 图五: clean 36-well observation series (sample 1506)
    S.USER_CFG.update(
        ob_file="dataset/v1/test/obData/Head1506.npz",
        feature="h_abs", sensor_ids=None, inset_clean_file=None,
        out_path=f"{OUT}/fig05_obs_series_clean.png",
    )
    S.main()
    # 图十: noisy 36-well series + clean-vs-noisy zoom inset (sample 1506)
    S.USER_CFG.update(
        ob_file="dataset/v1/test/obData_s36_noise_s003/Head1506.npz",
        feature="h_abs", sensor_ids=None,
        inset_clean_file="dataset/v1/test/obData/Head1506.npz",
        out_path=f"{OUT}/fig10_obs_series_noisy.png",
    )
    S.main()


def run_compare() -> None:
    from src.figs_paper import fig8_9_11_compare as C
    base = dict(
        truth_field_dirs=["dataset/v1/test/fields"], field_kind="auto",
        pred_file_name="lnK_mean.npy", grid_size=50, figure_dpi=300,
    )

    def _run(infer_runs, sample_id, out_name):
        C.USER_CFG.clear()
        C.USER_CFG.update(base, infer_runs=infer_runs, sample_id=sample_id,
                          out_path=f"{OUT}/{out_name}")
        C.main()

    # 图八: monitoring-well density (2x2), sample 1506
    _run(
        [
            {"name": "36-well estimate", "dir": "outputs/infer/exp_v1_baseline/v1"},
            {"name": "18-well estimate", "dir": "outputs/infer/exp_v1_s18/v1_s18"},
            {"name": "4-well estimate", "dir": "outputs/infer/exp_v1_s04/v1"},
        ],
        1506, "fig08_density_compare_id1506.png",
    )
    # 图九: full vs unconditional (1x3), samples 1501/1505/1507
    for sid in (1501, 1505, 1507):
        _run(
            [
                {"name": "WL-CDiff estimate", "dir": "outputs/infer/ablation/full"},
                {"name": "Unconditional output", "dir": "outputs/infer/ablation/E5_uncond"},
            ],
            sid, f"fig09_ablation_id{sid}.png",
        )
    # 图十一: normal vs noisy input (1x3), samples 1507/1553/1555
    for sid in (1507, 1553, 1555):
        _run(
            [
                {"name": "Noiseless estimate", "dir": "outputs/infer/exp_v1_baseline/v1"},
                {"name": "Noisy estimate", "dir": "outputs/infer/exp_v1_baseline/v1_noise_s003"},
            ],
            sid, f"fig11_noise_id{sid}.png",
        )


def run_typical() -> None:
    # 图六: typical reference/estimated cases for the full model, IDs reverse-found by SSIM
    from src.figs_paper import fig6_typical as T
    T.USER_CFG.update(out_path=f"{OUT}/fig06_typical_baseline.png")
    T.main()


def main() -> None:
    run_layout()
    run_series()
    run_compare()
    run_typical()


if __name__ == "__main__":
    main()
