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

from src.data.dataset import SensorDataset, denormalize_lnk
from src.models.diffusion import DiffusionScheduler, build_model_from_config
from src.run_config import ACTIVE, RUNTIME
from src.utils.io import ensure_dir, load_yaml
from src.utils.metrics import mae, rmse, ssim
from src.utils.runtime_paths import checkpoint_path_for_experiment, get_data_pair, get_experiment, processed_dir_for_pair


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


@torch.no_grad()
def main() -> None:
    rcfg = RUNTIME["infer_one"]
    exp_key, exp_cfg = get_experiment(rcfg.get("experiment"))
    train_pair_key, train_pair_cfg = get_data_pair(exp_cfg["train_data_pair"])
    test_pair_name = rcfg.get("test_data_pair") or ACTIVE.get("infer_test_pair") or ACTIVE["data_pair"]
    test_pair_key, test_pair_cfg = get_data_pair(test_pair_name)

    config_path = rcfg.get("config") or exp_cfg.get("config")
    cfg = load_yaml(config_path)
    device = resolve_device(rcfg.get("device") or cfg["project"]["device"])

    split_h5 = rcfg.get("split_h5") or str(processed_dir_for_pair(test_pair_key, test_pair_cfg) / "test.h5")
    stats_json = rcfg.get("stats_json") or str(processed_dir_for_pair(train_pair_key, train_pair_cfg) / "stats.json")
    ds = SensorDataset(split_h5, stats_json, normalize=True)
    sensor_xy = torch.from_numpy(ds.sensor_xy_norm).float().to(device)

    model = build_model_from_config(cfg).to(device).eval()
    ckpt_path = rcfg.get("checkpoint") or str(checkpoint_path_for_experiment(exp_key, exp_cfg))
    ckpt = torch.load(ckpt_path, map_location=device)
    use_ema = bool(rcfg.get("use_ema", False))
    if use_ema and "ema_model" in ckpt:
        model.load_state_dict(ckpt["ema_model"])
    else:
        model.load_state_dict(ckpt["model"])

    diffusion = DiffusionScheduler(timesteps=int(cfg["diffusion"]["timesteps"])).to(device)
    grid_size = int(cfg["data"]["grid_size"])
    index = int(rcfg.get("index", 0))
    num_samples = int(rcfg.get("num_samples") or cfg["inference"]["num_samples"])
    ddim_steps = int(rcfg.get("ddim_steps") or cfg["inference"]["ddim_steps"])
    cfg_scale = float(rcfg.get("cfg_scale") or cfg["inference"]["cfg_scale"])
    out_dir = rcfg.get("out_dir") or f"outputs/infer/{exp_key}/{test_pair_key}/one"

    print(
        f"[Config] experiment={exp_key} train_data_pair={train_pair_key} "
        f"test_data_pair={test_pair_key} ckpt={ckpt_path} index={index}"
    )

    sample = ds[index]
    obs = sample["obs_feat"].unsqueeze(0).to(device)  # (1,S,T,3)
    true_lnk = sample["lnk_raw"].squeeze(0).numpy()

    obs_rep = obs.repeat(num_samples, 1, 1, 1)
    samples_norm = diffusion.ddim_sample(
        model=model,
        obs_feat=obs_rep,
        sensor_xy=sensor_xy,
        grid_size=grid_size,
        num_steps=ddim_steps,
        cfg_scale=cfg_scale,
    )
    samples = denormalize_lnk(samples_norm, ds.stats).cpu().numpy().squeeze(1)  # (Ns,H,W)

    mean_map = samples.mean(axis=0)
    std_map = samples.std(axis=0)
    k_mean = np.exp(mean_map)

    out_dir = ensure_dir(out_dir)
    np.save(Path(out_dir) / "lnK_samples.npy", samples)
    np.save(Path(out_dir) / "lnK_mean.npy", mean_map)
    np.save(Path(out_dir) / "lnK_std.npy", std_map)
    np.save(Path(out_dir) / "K_mean.npy", k_mean)

    print(f"[OK] saved outputs to: {out_dir}")
    print(f"lnK RMSE={rmse(mean_map, true_lnk):.6f}, MAE={mae(mean_map, true_lnk):.6f}, SSIM={ssim(mean_map, true_lnk):.6f}")


if __name__ == "__main__":
    main()
