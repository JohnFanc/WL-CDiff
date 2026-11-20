from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from pathlib import Path

import h5py
import numpy as np

from src.data.sensor_utils import collect_field_by_id, parse_sample_id, read_lnk_file
from src.utils.io import ensure_dir, save_json


def save_split_h5(
    out_path: Path,
    obs_feat: np.ndarray,
    lnk: np.ndarray,
    sample_ids: np.ndarray,
    sensor_xy: np.ndarray,
    times: np.ndarray,
) -> None:
    ensure_dir(out_path.parent)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("obs_feat", data=obs_feat.astype(np.float32), compression="gzip")
        f.create_dataset("lnk", data=lnk.astype(np.float32), compression="gzip")
        f.create_dataset("sample_id", data=sample_ids.astype(np.int32))
        f.create_dataset("sensor_xy", data=sensor_xy.astype(np.int32))
        f.create_dataset("times", data=times.astype(np.float32))


def load_ob_dir(ob_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      obs_feat: (N,S,T,3)
      sample_ids: (N,)
      sensor_xy: (S,2)
      times: (T,)
    """
    ob_dir = Path(ob_dir)
    if not ob_dir.exists():
        raise FileNotFoundError(f"obData directory not found: {ob_dir}")

    files = sorted([p for p in ob_dir.glob("*.npz") if p.name.lower() != "meta.npz"])
    if not files:
        raise ValueError(f"No .npz files found in obData dir: {ob_dir}")

    obs_list: list[np.ndarray] = []
    sid_list: list[int] = []
    sensor_xy_ref = None
    times_ref = None

    for fp in files:
        data = np.load(fp)
        obs_feat = np.asarray(data["obs_feat"], dtype=np.float32)  # (S,T,3)
        sample_id = int(data["sample_id"]) if "sample_id" in data else parse_sample_id(fp)
        sensor_xy = np.asarray(data["sensor_xy"], dtype=np.int32)
        times = np.asarray(data["times"], dtype=np.float32)

        if sensor_xy_ref is None:
            sensor_xy_ref = sensor_xy
            times_ref = times
        else:
            if sensor_xy.shape != sensor_xy_ref.shape or not np.all(sensor_xy == sensor_xy_ref):
                raise ValueError(f"sensor_xy mismatch in {fp}")
            if times.shape != times_ref.shape or not np.allclose(times, times_ref):
                raise ValueError(f"times mismatch in {fp}")

        obs_list.append(obs_feat)
        sid_list.append(sample_id)

    ids = np.asarray(sid_list, dtype=np.int32)
    obs = np.stack(obs_list, axis=0)  # (N,S,T,3)

    # Stable ordering by sample id.
    idx = np.argsort(ids)
    return obs[idx], ids[idx], sensor_xy_ref, times_ref


def match_obs_with_fields(
    obs_feat: np.ndarray,
    sample_ids: np.ndarray,
    field_by_id: dict[int, str],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_idx = []
    lnk_list = []
    for i, sid in enumerate(sample_ids):
        field_path = field_by_id.get(int(sid))
        if field_path is None:
            continue
        keep_idx.append(i)
        lnk_list.append(read_lnk_file(field_path, grid_size=grid_size))

    if not keep_idx:
        raise ValueError("No matched <obData, field> pairs found.")

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    obs_out = obs_feat[keep_idx_arr]
    sid_out = sample_ids[keep_idx_arr]
    lnk_out = np.stack(lnk_list, axis=0)
    return obs_out, lnk_out, sid_out


def split_train_val(
    obs: np.ndarray,
    lnk: np.ndarray,
    sid: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n = obs.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return (obs[tr_idx], lnk[tr_idx], sid[tr_idx]), (obs[val_idx], lnk[val_idx], sid[val_idx])


def compute_stats(train_obs_feat: np.ndarray, train_lnk: np.ndarray) -> dict:
    obs_mean = train_obs_feat.mean(axis=(0, 1, 2))
    obs_std = train_obs_feat.std(axis=(0, 1, 2)) + 1e-6
    lnk_mean = float(train_lnk.mean())
    lnk_std = float(train_lnk.std() + 1e-6)
    return {
        "obs_mean": obs_mean.tolist(),
        "obs_std": obs_std.tolist(),
        "lnk_mean": lnk_mean,
        "lnk_std": lnk_std,
    }


def build_train_val_cache(
    train_ob_dir: str | Path,
    train_field_dir: str | Path,
    out_dir: str | Path,
    *,
    grid_size: int = 50,
    num_steps: int = 51,
    val_ratio: float = 0.1,
    seed: int = 42,
    field_kind: str = "auto",
) -> dict[str, str]:
    """
    Build train/val cache files and stats from one training obData + truth directory.
    Returns paths: train_h5, val_h5, stats_json
    """
    out_dir = Path(out_dir)
    train_ob_dir = Path(train_ob_dir)
    train_field_dir = Path(train_field_dir)

    obs_feat, sample_ids, sensor_xy, times = load_ob_dir(train_ob_dir)
    if times.shape[0] != int(num_steps):
        raise ValueError(f"num_steps mismatch: times={times.shape[0]} expected={num_steps}")

    field_map = collect_field_by_id([str(train_field_dir)], field_kind=field_kind)
    obs, lnk, sid = match_obs_with_fields(obs_feat, sample_ids, field_map, grid_size)

    (tr_obs, tr_lnk, tr_ids), (val_obs, val_lnk, val_ids) = split_train_val(obs, lnk, sid, val_ratio, seed)

    stats = compute_stats(tr_obs, tr_lnk)
    stats.update(
        {
            "grid_size": int(grid_size),
            "num_steps": int(num_steps),
            "num_sensors": int(sensor_xy.shape[0]),
            "field_kind": field_kind,
            "train_ob_dir": str(train_ob_dir),
            "train_field_dir": str(train_field_dir),
            "val_ratio": float(val_ratio),
            "split_seed": int(seed),
        }
    )

    train_h5 = out_dir / "train.h5"
    val_h5 = out_dir / "val.h5"
    stats_json = out_dir / "stats.json"

    save_split_h5(train_h5, tr_obs, tr_lnk, tr_ids, sensor_xy, times)
    save_split_h5(val_h5, val_obs, val_lnk, val_ids, sensor_xy, times)
    save_json(stats, stats_json)

    return {
        "train_h5": str(train_h5),
        "val_h5": str(val_h5),
        "stats_json": str(stats_json),
    }


def load_test_arrays(
    test_ob_dir: str | Path,
    test_field_dir: str | Path,
    *,
    grid_size: int = 50,
    num_steps: int = 51,
    field_kind: str = "auto",
) -> dict[str, np.ndarray]:
    """
    Load test arrays in-memory for inference/visualization.
    Returns: obs_feat(N,S,T,3), lnk(N,H,W), sample_id(N,), sensor_xy(S,2), times(T,)
    """
    test_ob_dir = Path(test_ob_dir)
    test_field_dir = Path(test_field_dir)

    obs_feat, sample_ids, sensor_xy, times = load_ob_dir(test_ob_dir)
    if times.shape[0] != int(num_steps):
        raise ValueError(f"num_steps mismatch: times={times.shape[0]} expected={num_steps}")

    field_map = collect_field_by_id([str(test_field_dir)], field_kind=field_kind)
    obs, lnk, sid = match_obs_with_fields(obs_feat, sample_ids, field_map, grid_size)
    return {
        "obs_feat": obs.astype(np.float32),
        "lnk": lnk.astype(np.float32),
        "sample_id": sid.astype(np.int32),
        "sensor_xy": sensor_xy.astype(np.int32),
        "times": times.astype(np.float32),
    }


USER_CFG = {
    # Optional standalone usage (not required for train/infer scripts).
    "train_ob_dir": "dataset/v1/train/obData",
    "train_field_dir": "dataset/v1/train/fields",
    "out_dir": "outputs/tmp_cache_train_v1",
    "grid_size": 50,
    "num_steps": 51,
    "val_ratio": 0.1,
    "seed": 42,
    "field_kind": "auto",
}


def main() -> None:
    result = build_train_val_cache(
        train_ob_dir=USER_CFG["train_ob_dir"],
        train_field_dir=USER_CFG["train_field_dir"],
        out_dir=USER_CFG["out_dir"],
        grid_size=int(USER_CFG["grid_size"]),
        num_steps=int(USER_CFG["num_steps"]),
        val_ratio=float(USER_CFG["val_ratio"]),
        seed=int(USER_CFG["seed"]),
        field_kind=str(USER_CFG["field_kind"]),
    )
    print(f"[OK] train_h5 -> {result['train_h5']}")
    print(f"[OK] val_h5   -> {result['val_h5']}")
    print(f"[OK] stats    -> {result['stats_json']}")


if __name__ == "__main__":
    main()
