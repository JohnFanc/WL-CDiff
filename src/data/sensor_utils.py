from __future__ import annotations

import csv
import re
from pathlib import Path

import h5py
import numpy as np


ID_PATTERN = re.compile(r"(\d+)")


def parse_sample_id(path: str | Path) -> int:
    stem = Path(path).stem
    m = ID_PATTERN.search(stem)
    if m is None:
        raise ValueError(f"Cannot parse sample id from: {path}")
    return int(m.group(1))


def load_sensor_config(csv_path: str | Path) -> np.ndarray:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["x"]), int(row["y"])))
    if not rows:
        raise ValueError(f"Empty sensor config: {csv_path}")
    return np.asarray(rows, dtype=np.int64)


def normalize_sensor_xy(sensor_xy: np.ndarray, grid_size: int) -> np.ndarray:
    return sensor_xy.astype(np.float32) / float(grid_size - 1)


def read_head_file(path: str | Path, grid_size: int = 50) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        values = np.asarray(f["Datasets/Head/Values"], dtype=np.float32)
        times = np.asarray(f["Datasets/Head/Times"], dtype=np.float32)

    if values.ndim == 2 and values.shape[1] == grid_size * grid_size:
        head = values.reshape(values.shape[0], grid_size, grid_size)
    elif values.ndim == 3 and values.shape[1:] == (grid_size, grid_size):
        head = values
    else:
        raise ValueError(f"Unexpected Head/Values shape {values.shape} in {path}")
    return head, times


def read_lnk_file(path: str | Path, grid_size: int = 50) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.shape != (grid_size, grid_size):
        if arr.size == grid_size * grid_size:
            arr = arr.reshape(grid_size, grid_size)
        else:
            raise ValueError(f"Unexpected lnK/K shape {arr.shape} in {path}")

    name = path.name.lower()
    # Accept both lnKfield_* and Kfield_* truth files.
    if name.startswith("kfield_") and not name.startswith("lnkfield_"):
        arr = np.log(np.clip(arr, 1e-12, None))
    elif not (name.startswith("lnkfield_") or name.startswith("kfield_")):
        raise ValueError(f"Unexpected lnK shape {arr.shape} in {path}")
    return arr


def extract_sensor_series(head_3d: np.ndarray, sensor_xy: np.ndarray) -> np.ndarray:
    if head_3d.ndim != 3:
        raise ValueError(f"head_3d should be (T,H,W), got {head_3d.shape}")
    t, h, w = head_3d.shape
    out = np.zeros((sensor_xy.shape[0], t), dtype=np.float32)
    for i, (x, y) in enumerate(sensor_xy):
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError(f"Sensor {(x, y)} is outside grid {(h, w)}")
        out[i] = head_3d[:, y, x]
    return out


def build_obs_features(obs_abs: np.ndarray, times: np.ndarray | None = None) -> np.ndarray:
    if obs_abs.ndim != 2:
        raise ValueError(f"obs_abs should be (S,T), got {obs_abs.shape}")
    steady = obs_abs[:, :1]
    delta = obs_abs - steady
    if times is None:
        dhdt = np.gradient(obs_abs, axis=1)
    else:
        dhdt = np.gradient(obs_abs, times.astype(np.float32), axis=1)
    feat = np.stack([obs_abs, delta, dhdt], axis=-1)
    return feat.astype(np.float32)


def _field_priority(path: Path) -> int:
    n = path.name.lower()
    if n.startswith("lnkfield_"):
        return 2
    if n.startswith("kfield_"):
        return 1
    return 0


def _collect_lnk_files(paths_or_dirs: list[str], field_kind: str = "auto") -> dict[int, str]:
    if field_kind not in {"auto", "lnk", "k"}:
        raise ValueError(f"field_kind must be one of auto/lnk/k, got: {field_kind}")

    out: dict[int, str] = {}
    for entry in paths_or_dirs:
        p = Path(entry)
        if p.is_dir():
            cand: list[Path] = []
            if field_kind in {"auto", "lnk"}:
                cand += list(p.glob("lnKfield_*.csv")) + list(p.glob("lnKfield_*.npy"))
            if field_kind in {"auto", "k"}:
                cand += list(p.glob("Kfield_*.csv")) + list(p.glob("Kfield_*.npy"))
        else:
            cand = list(Path().glob(entry))
        for f in cand:
            sid = parse_sample_id(f)
            if sid not in out:
                out[sid] = str(f)
            elif _field_priority(Path(out[sid])) < _field_priority(f):
                # In auto mode, prefer lnKfield over Kfield when both exist.
                out[sid] = str(f)
    return out


def collect_field_by_id(field_sources: list[str], field_kind: str = "auto") -> dict[int, str]:
    return _collect_lnk_files(field_sources, field_kind=field_kind)


def pair_head_lnk(head_paths: list[str], lnk_sources: list[str], field_kind: str = "auto") -> list[tuple[int, str, str]]:
    lnk_by_id = _collect_lnk_files(lnk_sources, field_kind=field_kind)
    pairs: list[tuple[int, str, str]] = []
    for h in head_paths:
        sid = parse_sample_id(h)
        lnk = lnk_by_id.get(sid)
        if lnk is not None:
            pairs.append((sid, h, lnk))
    pairs.sort(key=lambda x: x[0])
    return pairs
