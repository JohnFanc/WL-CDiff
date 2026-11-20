from __future__ import annotations

from pathlib import Path

from src.run_config import ACTIVE, DATA_PAIRS, EXPERIMENTS


def get_data_pair_key(override: str | None = None) -> str:
    key = override or ACTIVE["data_pair"]
    if key not in DATA_PAIRS:
        raise KeyError(f"Unknown data pair: {key}")
    return key


def get_data_pair(override: str | None = None) -> tuple[str, dict]:
    key = get_data_pair_key(override)
    return key, DATA_PAIRS[key]


def get_experiment_key(override: str | None = None) -> str:
    key = override or ACTIVE["experiment"]
    if key not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {key}")
    return key


def get_experiment(override: str | None = None) -> tuple[str, dict]:
    key = get_experiment_key(override)
    return key, EXPERIMENTS[key]


def processed_dir_for_pair(pair_key: str, pair_cfg: dict | None = None) -> Path:
    cfg = pair_cfg or DATA_PAIRS[pair_key]
    return Path(cfg.get("processed_dir", f"dataset/processed/{pair_key}"))


def split_dir_for_pair(pair_cfg: dict, split: str) -> Path:
    if split == "train":
        return Path(pair_cfg["train_split_dir"])
    if split == "test":
        return Path(pair_cfg["test_split_dir"])
    raise ValueError(f"split must be train or test, got: {split}")


def obdata_dir_for_pair(pair_cfg: dict, split: str) -> Path:
    return split_dir_for_pair(pair_cfg, split) / pair_cfg["obdata_subdir"]


def field_dir_for_pair(pair_cfg: dict, split: str) -> Path:
    return split_dir_for_pair(pair_cfg, split) / pair_cfg["fields_subdir"]


def heads_dir_for_pair(pair_cfg: dict, split: str) -> Path:
    return split_dir_for_pair(pair_cfg, split) / pair_cfg["heads_subdir"]


def train_output_dir_for_experiment(exp_key: str, exp_cfg: dict | None = None) -> Path:
    cfg = exp_cfg or EXPERIMENTS[exp_key]
    return Path(cfg.get("train_output_dir", f"outputs/train/{exp_key}"))


def checkpoint_path_for_experiment(
    exp_key: str, exp_cfg: dict | None = None, checkpoint_name: str | None = None
) -> Path:
    cfg = exp_cfg or EXPERIMENTS[exp_key]
    ckpt_name = checkpoint_name or cfg.get("checkpoint_name", "best.pt")
    return train_output_dir_for_experiment(exp_key, cfg) / "checkpoints" / ckpt_name


def infer_output_dir(exp_key: str, test_pair_key: str, exp_cfg: dict | None = None) -> Path:
    cfg = exp_cfg or EXPERIMENTS[exp_key]
    root = Path(cfg.get("infer_output_root", f"outputs/infer/{exp_key}"))
    return root / test_pair_key

