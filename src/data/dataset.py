from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.sensor_utils import normalize_sensor_xy
from src.utils.io import load_json


def denormalize_lnk(x_norm: torch.Tensor, stats: dict) -> torch.Tensor:
    return x_norm * float(stats["lnk_std"]) + float(stats["lnk_mean"])


class SensorDataset(Dataset):
    def __init__(self, h5_path: str | Path, stats_path: str | Path, normalize: bool = True) -> None:
        self.h5_path = Path(h5_path)
        self.stats = load_json(stats_path)
        self.normalize = normalize

        with h5py.File(self.h5_path, "r") as f:
            self.obs_feat = np.asarray(f["obs_feat"], dtype=np.float32)  # (N,S,T,3)
            self.lnk = np.asarray(f["lnk"], dtype=np.float32)  # (N,H,W)
            self.sample_id = np.asarray(f["sample_id"], dtype=np.int32)
            self.sensor_xy = np.asarray(f["sensor_xy"], dtype=np.int64)  # (S,2)
            self.times = np.asarray(f["times"], dtype=np.float32)

        grid_size = int(self.stats.get("grid_size", 50))
        self.sensor_xy_norm = normalize_sensor_xy(self.sensor_xy, grid_size)

        self.obs_mean = np.asarray(self.stats["obs_mean"], dtype=np.float32)  # (3,)
        self.obs_std = np.asarray(self.stats["obs_std"], dtype=np.float32)  # (3,)
        self.lnk_mean = float(self.stats["lnk_mean"])
        self.lnk_std = float(self.stats["lnk_std"])

    def __len__(self) -> int:
        return self.obs_feat.shape[0]

    def _normalize_obs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.obs_mean[None, None, :]) / self.obs_std[None, None, :]

    def _normalize_lnk(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lnk_mean) / self.lnk_std

    def __getitem__(self, idx: int) -> dict:
        obs = self.obs_feat[idx]  # (S,T,3)
        lnk_raw = self.lnk[idx]  # (H,W)

        if self.normalize:
            obs_in = self._normalize_obs(obs)
            lnk_in = self._normalize_lnk(lnk_raw)
        else:
            obs_in = obs
            lnk_in = lnk_raw

        return {
            "obs_feat": torch.from_numpy(obs_in).float(),
            "lnk": torch.from_numpy(lnk_in).unsqueeze(0).float(),
            "lnk_raw": torch.from_numpy(lnk_raw).unsqueeze(0).float(),
            "sample_id": int(self.sample_id[idx]),
        }
