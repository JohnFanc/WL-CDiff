from __future__ import annotations

"""
Per-pathway ablation evaluation for the conditioning module.

Runs inference (reusing src.infer.infer_and_viz) for each ablation checkpoint on
the 36-well test set, then aggregates the mean RMSE/MAE/SSIM (lnK & K) of each
into outputs/infer/ablation/summary_ablation.csv.

Run (in the working torch env), from project root the project root:
  python -m src.infer.run_ablation_eval
"""

import sys
import csv
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch  # noqa: E402
import src.infer.infer_and_viz as iv  # noqa: E402

# Skip figure generation (slow); we only need metrics + npy arrays.
_noop = lambda *a, **k: None  # noqa: E731
iv.save_lnk_pair_with_metrics = _noop
iv.save_uncertainty_map = _noop
iv.save_triplet = _noop

EXPS = [
    ("full",          "exp_v1_baseline"),
    ("E2_no_condmap", "exp_v1_s36_ablation_E2"),
    ("E3_no_global",  "exp_v1_s36_ablation_E3"),
    ("E4_no_crossattn", "exp_v1_s36_ablation_E4"),
    ("E5_uncond",     "exp_v1_s36_ablation_E5"),
]

TEST_OB = "dataset/v1/test/obData"
TEST_FIELDS = "dataset/v1/test/fields"


def read_summary(csv_path: Path) -> dict:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if str(r.get("sample_id")) == "-1":
            return r
    return rows[-1]


def main() -> None:
    out_dir = Path("outputs/infer/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []

    for name, exp in EXPS:
        torch.manual_seed(42)
        iv.USER_CFG.update(
            checkpoint_path=f"outputs/train/{exp}/checkpoints/best.pt",
            test_obdata_dir=TEST_OB,
            test_field_truth_dir=TEST_FIELDS,
            infer_output_dir=f"outputs/infer/ablation/{name}/",
            use_ema=True,
            num_samples=None,
            ddim_steps=None,
            cfg_scale=None,
            max_items=None,
        )
        print(f"\n===== RUN {name} ({exp}) =====", flush=True)
        iv.main()
        s = read_summary(Path(f"outputs/infer/ablation/{name}/metrics_log.csv"))
        out_rows.append({
            "name": name, "exp": exp,
            "rmse_lnk": s["rmse_lnk"], "mae_lnk": s["mae_lnk"], "ssim_lnk": s["ssim_lnk"],
            "rmse_k": s["rmse_k"], "mae_k": s["mae_k"], "ssim_k": s["ssim_k"],
        })

    with open(out_dir / "summary_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    print("\n===== ABLATION SUMMARY (mean over test set) =====", flush=True)
    for r in out_rows:
        print(
            f"{r['name']:16s} | lnK RMSE={float(r['rmse_lnk']):.4f} MAE={float(r['mae_lnk']):.4f} "
            f"SSIM={float(r['ssim_lnk']):.4f} || K RMSE={float(r['rmse_k']):.4f} "
            f"MAE={float(r['mae_k']):.4f} SSIM={float(r['ssim_k']):.4f}", flush=True)
    print("[OK] ->", out_dir / "summary_ablation.csv", flush=True)


if __name__ == "__main__":
    main()
