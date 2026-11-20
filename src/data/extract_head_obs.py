from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from pathlib import Path

import numpy as np

from src.data.sensor_utils import (
    build_obs_features,
    extract_sensor_series,
    load_sensor_config,
    parse_sample_id,
    read_head_file,
)
from src.utils.io import ensure_dir, save_json


USER_CFG = {
    # Input: directory containing Head*.h5 files.
    "head_dir": "dataset/v1/train/heads",
    "head_glob": "Head*.h5",
    # Sensor configuration.
    "sensors_csv": "configs/sensors_04.csv",
    "grid_size": 50,
    "num_steps": 51,
    # Output: directory for extracted observation sequences (.npz).
    "obdata_out_dir": "dataset/v1/train/obData_s04",
    # Noise switch and parameters.
    "noise_enable": False,
    "noise_type": "gaussian",  # currently supported: gaussian
    "noise_sigma_head_m": 0.03,
    "noise_seed": 42,  # effective seed = noise_seed + sample_id
    "noise_clip_min_head_m": None,
    "noise_clip_max_head_m": None,
}


def save_one_ob_file(
    out_path: str | Path,
    sample_id: int,
    obs_abs: np.ndarray,
    obs_feat: np.ndarray,
    sensor_xy: np.ndarray,
    times: np.ndarray,
    head_file: str,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        sample_id=np.int32(sample_id),
        obs_abs=obs_abs.astype(np.float32),  # (S,T)
        obs_feat=obs_feat.astype(np.float32),  # (S,T,3)
        sensor_xy=sensor_xy.astype(np.int32),  # (S,2)
        times=times.astype(np.float32),  # (T,)
        head_file=np.asarray(head_file),
    )


def _add_observation_noise(obs_abs: np.ndarray, sample_id: int, cfg: dict) -> np.ndarray:
    if not bool(cfg.get("noise_enable", False)):
        return obs_abs

    noise_type = str(cfg.get("noise_type", "gaussian")).lower()
    if noise_type != "gaussian":
        raise ValueError(f"Unsupported noise_type: {noise_type}. Expected gaussian.")

    sigma = float(cfg.get("noise_sigma_head_m", 0.0))
    if sigma <= 0.0:
        return obs_abs

    seed = int(cfg.get("noise_seed", 42))
    rng = np.random.default_rng(seed + int(sample_id))
    noise = rng.normal(loc=0.0, scale=sigma, size=obs_abs.shape).astype(np.float32)
    out = obs_abs + noise

    clip_min = cfg.get("noise_clip_min_head_m")
    clip_max = cfg.get("noise_clip_max_head_m")
    if clip_min is not None or clip_max is not None:
        out = np.clip(out, clip_min, clip_max)
    return out.astype(np.float32)


def extract_observations(cfg: dict) -> tuple[int, Path]:
    head_dir = Path(cfg["head_dir"])
    obdata_out_dir = Path(cfg["obdata_out_dir"])
    head_glob = str(cfg.get("head_glob", "Head*.h5"))
    sensors_csv = str(cfg["sensors_csv"])
    grid_size = int(cfg["grid_size"])
    num_steps = int(cfg["num_steps"])

    sensor_xy = load_sensor_config(sensors_csv)
    head_files = sorted(head_dir.glob(head_glob))
    if not head_files:
        raise ValueError(f"No head files found: {head_dir}/{head_glob}")

    ensure_dir(obdata_out_dir)
    count = 0

    for hp in head_files:
        sid = parse_sample_id(hp)
        head, times = read_head_file(hp, grid_size=grid_size)
        if head.shape[0] != num_steps:
            raise ValueError(f"Unexpected steps in {hp}: {head.shape[0]} != {num_steps}")

        obs_abs_clean = extract_sensor_series(head, sensor_xy)  # (S,T)
        obs_abs = _add_observation_noise(obs_abs_clean, sid, cfg)
        obs_feat = build_obs_features(obs_abs, times)  # (S,T,3)

        out_file = obdata_out_dir / f"{hp.stem}.npz"
        save_one_ob_file(
            out_path=out_file,
            sample_id=sid,
            obs_abs=obs_abs,
            obs_feat=obs_feat,
            sensor_xy=sensor_xy,
            times=times,
            head_file=hp.name,
        )
        count += 1

    meta = {
        "head_dir": str(head_dir),
        "head_glob": head_glob,
        "obdata_out_dir": str(obdata_out_dir),
        "num_files": count,
        "num_sensors": int(sensor_xy.shape[0]),
        "num_steps": num_steps,
        "grid_size": grid_size,
        "sensors_csv": sensors_csv,
        "noise_enable": bool(cfg.get("noise_enable", False)),
        "noise_type": cfg.get("noise_type", "gaussian"),
        "noise_sigma_head_m": float(cfg.get("noise_sigma_head_m", 0.0)),
        "noise_seed": int(cfg.get("noise_seed", 42)),
        "noise_clip_min_head_m": cfg.get("noise_clip_min_head_m"),
        "noise_clip_max_head_m": cfg.get("noise_clip_max_head_m"),
    }
    save_json(meta, obdata_out_dir / "meta.json")
    return count, obdata_out_dir


def main() -> None:
    n, out_dir = extract_observations(dict(USER_CFG))
    print(f"[OK] extracted {n} files -> {out_dir}")


if __name__ == "__main__":
    main()
