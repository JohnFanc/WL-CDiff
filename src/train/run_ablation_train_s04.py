from __future__ import annotations

"""
Train per-pathway ablation models under the sparse 4-well setting (s04).

For each of E2/E3/E4, copies configs/default.yaml, patches the model.ablation
block, writes a temp config, and runs the existing training loop (src.train.train)
on dataset/v1/train/obData_s04 into outputs/train/exp_v1_s04_ablation_<E>.

The full 4-well model (exp_v1_s04) already exists and is NOT retrained here.

Run (working torch env), from project root the project root:
  python -m src.train.run_ablation_train_s04
Smoke test (2 epochs, throwaway dirs):
  set SMOKE=1 && python -m src.train.run_ablation_train_s04
"""

import os
import sys
import copy
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import yaml  # noqa: E402
import src.train.train as tr  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"

# Exact flags matching the 36-well single-pathway ablations (E2/E3/E4).
VARIANTS = {
    "E2": dict(disable_token_to_cond_map=True,  global_broadcast_channels=0,
               disable_global_to_adagn=False, disable_sensor_mid_cross_attn=False),
    "E3": dict(disable_token_to_cond_map=False, global_broadcast_channels=0,
               disable_global_to_adagn=True,  disable_sensor_mid_cross_attn=False),
    "E4": dict(disable_token_to_cond_map=False, global_broadcast_channels=0,
               disable_global_to_adagn=False, disable_sensor_mid_cross_attn=True),
}


def main() -> None:
    with open("configs/default.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    only = os.environ.get("ONLY", "").strip()
    if SMOKE:
        names = ["E2"]
    elif only:
        names = [x.strip() for x in only.split(",") if x.strip() in VARIANTS]
    else:
        names = ["E2", "E3", "E4"]
    if not names:
        raise SystemExit("No valid variant selected (ONLY=%r)" % only)
    suffix = "_smoke" if SMOKE else ""

    for name in names:
        cfg = copy.deepcopy(base)
        cfg["model"]["ablation"].update(VARIANTS[name])
        cfg_path = f"configs/_abl_s04_{name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        tr.USER_CFG.update(
            config_path=cfg_path,
            train_obdata_dir="dataset/v1/train/obData_s04",
            train_field_truth_dir="dataset/v1/train/fields",
            train_output_dir=f"outputs/train/exp_v1_s04_ablation_{name}{suffix}",
            experiment_name=f"exp_v1_s04_ablation_{name}{suffix}",
            auto_build_dataset_cache=True,
            dataset_cache_dir=None,
            epochs=(2 if SMOKE else None),
        )
        print(f"\n##### TRAIN {name} (smoke={SMOKE}) cfg={cfg_path} -> "
              f"outputs/train/exp_v1_s04_ablation_{name}{suffix} #####", flush=True)
        tr.main()

    print("\n[OK] ablation training finished:", names, flush=True)


if __name__ == "__main__":
    main()
