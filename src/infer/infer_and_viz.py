from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from src.data.build_sensor_dataset import load_test_arrays
from src.data.dataset import denormalize_lnk
from src.data.sensor_utils import normalize_sensor_xy
from src.models.diffusion import DiffusionScheduler, build_model_from_config
from src.utils.io import ensure_dir, load_json, load_yaml, write_csv
from src.utils.metrics import mae, rmse, ssim
from src.utils.viz import save_lnk_pair_with_metrics, save_triplet, save_uncertainty_map


USER_CFG = {
    # -----------------------------------------------------------------
    # Core paths: only these four are required for normal use.
    # -----------------------------------------------------------------
    "checkpoint_path": "outputs/train/exp_v1_s04/checkpoints/best.pt",
    "test_obdata_dir": "dataset/v1/test/obData_s04",
    "test_field_truth_dir": "dataset/v1/test/fields",
    "infer_output_dir": "outputs/infer/exp_v1_s04/v1/",
    # -----------------------------------------------------------------
    # Optional settings.
    # -----------------------------------------------------------------
    "config_path": "configs/default.yaml",
    "stats_json_fallback": None,  # only used if checkpoint has no "stats"
    "field_kind": "auto",  # auto | lnk | k
    "grid_size": 50,
    "num_steps": 51,
    "num_samples": None,  # None -> YAML default
    "ddim_steps": None,  # None -> YAML default
    "cfg_scale": None,  # None -> YAML default
    "max_items": None,
    "device": None,  # None -> YAML default
    "use_ema": True,
}


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def metric_row(pred_lnk: np.ndarray, true_lnk: np.ndarray) -> dict[str, float]:
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


def _infer_stats_candidates(ckpt: dict, ckpt_path: Path, test_obdata_dir: str) -> list[Path]:
    candidates: list[Path] = []

    # New training pipeline cache layout: <train_output_dir>/cache/processed/stats.json
    candidates.append(ckpt_path.parent.parent / "cache" / "processed" / "stats.json")
    # Optional sidecar stats near train output root.
    candidates.append(ckpt_path.parent.parent / "stats.json")

    # From checkpoint metadata when available.
    train_pair = ckpt.get("train_data_pair")
    if isinstance(train_pair, str) and train_pair:
        candidates.append(Path("dataset") / "processed" / train_pair / "stats.json")

    # Heuristic from obData path.
    ob_dir = Path(test_obdata_dir)
    ob_name = ob_dir.name.lower()
    if "s18" in ob_name:
        candidates.append(Path("dataset/processed/v1_s18/stats.json"))

    parts = [p.lower() for p in ob_dir.parts]
    if "dataset" in parts:
        i = parts.index("dataset")
        if i + 1 < len(parts):
            pair_like = ob_dir.parts[i + 1]
            candidates.append(Path("dataset") / "processed" / pair_like / "stats.json")

    # Common defaults.
    candidates.append(Path("dataset/processed/v1/stats.json"))
    candidates.append(Path("dataset/processed/v1_s18/stats.json"))

    # Deduplicate while preserving order.
    uniq: list[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _load_stats(ckpt: dict, ckpt_path: Path, fallback_path: str | None, test_obdata_dir: str) -> tuple[dict, str]:
    if "stats" in ckpt and isinstance(ckpt["stats"], dict):
        return ckpt["stats"], "checkpoint:stats"

    if fallback_path:
        p = Path(fallback_path)
        if p.exists():
            return load_json(p), str(p)

    candidates = _infer_stats_candidates(ckpt, ckpt_path, test_obdata_dir)
    for p in candidates:
        if p.exists():
            return load_json(p), str(p)

    searched = "\n".join([f"- {p}" for p in candidates])
    raise KeyError(
        "No normalization stats found. Provide USER_CFG['stats_json_fallback'] or make one of these paths available:\n"
        f"{searched}"
    )


def _normalize_obs(obs_feat: np.ndarray, stats: dict) -> np.ndarray:
    obs_mean = np.asarray(stats["obs_mean"], dtype=np.float32)  # (3,)
    obs_std = np.asarray(stats["obs_std"], dtype=np.float32)  # (3,)
    return (obs_feat - obs_mean[None, None, None, :]) / obs_std[None, None, None, :]


@torch.no_grad()
def main() -> None:
    ucfg = dict(USER_CFG)
    ckpt_path = Path(ucfg["checkpoint_path"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"] if isinstance(ckpt.get("config"), dict) else load_yaml(ucfg["config_path"])
    ab_cfg = cfg.get("model", {}).get("ablation", {})

    device = resolve_device(str(ucfg["device"] or cfg["project"]["device"]))
    grid_size = int(ucfg["grid_size"] or cfg["data"]["grid_size"])
    num_steps = int(ucfg["num_steps"] or cfg["data"]["num_steps"])
    num_samples = int(ucfg["num_samples"] or cfg["inference"]["num_samples"])
    ddim_steps = int(ucfg["ddim_steps"] or cfg["inference"]["ddim_steps"])
    cfg_scale = float(ucfg["cfg_scale"] or cfg["inference"]["cfg_scale"])
    max_items = ucfg.get("max_items")
    use_ema = bool(ucfg.get("use_ema", True))

    stats, stats_src = _load_stats(ckpt, ckpt_path, ucfg.get("stats_json_fallback"), str(ucfg["test_obdata_dir"]))

    model = build_model_from_config(cfg).to(device).eval()
    if use_ema and "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    else:
        model.load_state_dict(ckpt["model"])
    diffusion = DiffusionScheduler(timesteps=int(cfg["diffusion"]["timesteps"])).to(device)

    test = load_test_arrays(
        test_ob_dir=ucfg["test_obdata_dir"],
        test_field_dir=ucfg["test_field_truth_dir"],
        grid_size=grid_size,
        num_steps=num_steps,
        field_kind=str(ucfg["field_kind"]),
    )
    obs_feat_raw = test["obs_feat"]  # (N,S,T,3)
    true_lnk_all = test["lnk"]  # (N,H,W)
    sample_ids = test["sample_id"]  # (N,)
    sensor_xy = test["sensor_xy"]  # (S,2)
    obs_feat_norm = _normalize_obs(obs_feat_raw, stats).astype(np.float32)
    sensor_xy_norm = normalize_sensor_xy(sensor_xy, grid_size=grid_size)
    sensor_xy_t = torch.from_numpy(sensor_xy_norm).float().to(device)

    out_root = ensure_dir(ucfg["infer_output_dir"])
    arr_dir = ensure_dir(Path(out_root) / "arrays")
    fig_dir = ensure_dir(Path(out_root) / "figures")

    n_items = len(sample_ids) if max_items is None else min(len(sample_ids), int(max_items))
    rows = []

    print(
        f"[Config] ckpt={ckpt_path} test_obdata_dir={ucfg['test_obdata_dir']} "
        f"test_field_truth_dir={ucfg['test_field_truth_dir']} out_dir={out_root}"
    )
    print(
        "[Ablation] "
        f"disable_token_to_cond_map={bool(ab_cfg.get('disable_token_to_cond_map', False))} "
        f"global_broadcast_channels={int(ab_cfg.get('global_broadcast_channels', 8))} "
        f"disable_global_to_adagn={bool(ab_cfg.get('disable_global_to_adagn', False))} "
        f"disable_sensor_mid_cross_attn={bool(ab_cfg.get('disable_sensor_mid_cross_attn', False))}"
    )
    print(f"[Config] stats_source={stats_src}")

    for idx in range(n_items):
        sid = int(sample_ids[idx])
        obs = torch.from_numpy(obs_feat_norm[idx]).float().unsqueeze(0).to(device)
        true_lnk = true_lnk_all[idx].astype(np.float32)

        obs_rep = obs.repeat(num_samples, 1, 1, 1)
        samples_norm = diffusion.ddim_sample(
            model=model,
            obs_feat=obs_rep,
            sensor_xy=sensor_xy_t,
            grid_size=grid_size,
            num_steps=ddim_steps,
            cfg_scale=cfg_scale,
        )
        samples = denormalize_lnk(samples_norm, stats).cpu().numpy().squeeze(1)  # (Ns,H,W)
        pred_lnk = samples.mean(axis=0)
        std_lnk = samples.std(axis=0)
        pred_k = np.exp(pred_lnk)

        sample_arr_dir = ensure_dir(Path(arr_dir) / f"sample_{sid:04d}")
        np.save(sample_arr_dir / "lnK_samples.npy", samples)
        np.save(sample_arr_dir / "lnK_mean.npy", pred_lnk)
        np.save(sample_arr_dir / "lnK_std.npy", std_lnk)
        np.save(sample_arr_dir / "K_mean.npy", pred_k)

        true_k = np.exp(true_lnk)
        sample_fig_dir = ensure_dir(Path(fig_dir) / f"sample_{sid:04d}")
        m = metric_row(pred_lnk, true_lnk)
        save_lnk_pair_with_metrics(
            true_map=true_lnk,
            pred_map=pred_lnk,
            out_path=sample_fig_dir / "triplet_lnk.png",
            rmse_value=m["rmse_lnk"],
            mae_value=m["mae_lnk"],
            ssim_value=m["ssim_lnk"],
        )
        save_uncertainty_map(
            std_lnk,
            out_path=sample_fig_dir / "lnk_std.png",
            title="lnK Std",
        )
        save_triplet(
            true_k,
            pred_k,
            pred_k - true_k,
            out_path=sample_fig_dir / "triplet_k.png",
            title=f"K True-Pred-Diff | ID={sid}",
            value_cmap="plasma",
        )

        m["index"] = idx
        m["sample_id"] = sid
        rows.append(m)
        print(
            f"[{idx + 1}/{n_items}] ID={sid} "
            f"RMSE(lnK)={m['rmse_lnk']:.4f} MAE(lnK)={m['mae_lnk']:.4f} SSIM(lnK)={m['ssim_lnk']:.4f}"
        )

    if rows:
        summary = {"index": -1, "sample_id": -1}
        for key in rows[0]:
            if key in ("index", "sample_id"):
                continue
            summary[key] = float(np.mean([r[key] for r in rows]))
        rows.append(summary)
        csv_path = Path(out_root) / "metrics_log.csv"
        write_csv(csv_path, fieldnames=rows[0].keys(), rows=rows)
        print(f"[OK] metrics -> {csv_path}")


if __name__ == "__main__":
    main()
