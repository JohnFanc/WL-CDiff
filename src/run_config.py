from __future__ import annotations

"""
Central runtime configuration.

Use ACTIVE to quickly switch which data pair and experiment are used.
Add more entries in DATA_PAIRS / EXPERIMENTS for multi-dataset workflows.
"""

# ---------------------------------------------------------------------
# Active selection
# ---------------------------------------------------------------------
ACTIVE = {
    # The default data pair used by data preprocessing scripts.
    "data_pair": "v1_s18",
    # The default training experiment.
    "experiment": "exp_v1_s18",
    # Optional override for inference test pair. None -> use ACTIVE["data_pair"].
    "infer_test_pair": None,
}


# ---------------------------------------------------------------------
# Data pair registry
# Each data pair defines a train split and a test split.
# ---------------------------------------------------------------------
DATA_PAIRS = {
    "v1": {
        "sensors_csv": "configs/sensors_36.csv",
        "train_split_dir": "dataset/v1/train",
        "test_split_dir": "dataset/v1/test",
        "heads_subdir": "heads",
        "fields_subdir": "fields",
        "obdata_subdir": "obData",
        "field_kind": "lnk",  # auto | lnk | k
        "grid_size": 50,
        "num_steps": 51,
        "val_ratio": 0.1,
        "split_seed": 42,
        # Processed output namespace for this data pair.
        "processed_dir": "dataset/processed/v1",
    },
    "v1_s18": {
        "sensors_csv": "configs/sensors_18.csv",
        "train_split_dir": "dataset/v1/train",
        "test_split_dir": "dataset/v1/test",
        "heads_subdir": "heads",
        "fields_subdir": "fields",
        "obdata_subdir": "obData_s18",
        "field_kind": "lnk",  # auto | lnk | k
        "grid_size": 50,
        "num_steps": 51,
        "val_ratio": 0.1,
        "split_seed": 42,
        # Processed output namespace for this data pair.
        "processed_dir": "dataset/processed/v1_s18",
    }
}


# ---------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------
EXPERIMENTS = {
    "exp_v1_baseline": {
        "config": "configs/default.yaml",
        # Which data pair this model is trained on.
        "train_data_pair": "v1",
        # Training output namespace.
        "train_output_dir": "outputs/train/exp_v1_baseline",
        # Inference output namespace root; test pair will be appended.
        "infer_output_root": "outputs/infer/exp_v1_baseline",
        # Preferred checkpoint filename in train_output_dir/checkpoints.
        "checkpoint_name": "best.pt",
    },
    "exp_v1_s18": {
        "config": "configs/default.yaml",
        # Which data pair this model is trained on.
        "train_data_pair": "v1_s18",
        # Training output namespace.
        "train_output_dir": "outputs/train/exp_v1_s18",
        # Inference output namespace root; test pair will be appended.
        "infer_output_root": "outputs/infer/exp_v1_s18",
        # Preferred checkpoint filename in train_output_dir/checkpoints.
        "checkpoint_name": "best.pt",
    }
}


# ---------------------------------------------------------------------
# Script-level overrides
# Keep None to use ACTIVE + registry defaults.
# ---------------------------------------------------------------------
RUNTIME = {
    "extract_head_obs": {
        "data_pair": None,
        "split_dirs": None,  # optional explicit list of split dirs
    },
    "build_sensor_dataset": {
        "data_pair": None,
        "out_dir": None,
        "val_ratio": None,
        "seed": None,
        "field_kind": None,  # auto | lnk | k
    },
    "train": {
        "experiment": None,
        "config": None,
        "device": None,
        "epochs": None,
        "batch_size": None,
        "lr": None,
        "num_workers": None,
    },
    "infer_one": {
        "experiment": None,
        "test_data_pair": None,
        "config": None,
        "checkpoint": None,
        "stats_json": None,  # default uses train pair stats
        "split_h5": None,  # default uses selected test pair test.h5
        "index": 0,
        "num_samples": None,
        "ddim_steps": None,
        "cfg_scale": None,
        "device": None,
        "out_dir": None,
        "use_ema": True,
    },
    "infer_and_viz": {
        "experiment": None,
        "test_data_pair": None,
        "config": None,
        "checkpoint": None,
        "stats_json": None,  # default uses train pair stats
        "test_h5": None,  # default uses selected test pair test.h5
        "out_dir": None,
        "num_samples": None,
        "ddim_steps": None,
        "cfg_scale": None,
        "max_items": None,
        "device": None,
        "use_ema": True,
    },
    "viz_sensor_locations": {
        "data_pair": None,
        "config": None,
        "sensors_csv": None,
        "grid_size": None,
        "out_path": None,
        "show_sensor_id": True,
        "marker_size": 60.0,
    },
    "plot_ob_sequence": {
        "data_pair": None,
        "config": None,
        "ob_file": None,  # direct .npz file path in obData
        "split": "train",  # train | test
        "sensor_ids": None,
        "feature": "h_abs",  # h_abs | delta | dhdt
        "out_path": None,
        "sequence_name": None,
        "sample_id": None,
    },
}
