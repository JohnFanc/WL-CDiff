from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import copy
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.data.build_sensor_dataset import build_train_val_cache
from src.data.dataset import SensorDataset
from src.models.diffusion import DiffusionScheduler, build_model_from_config
from src.train.losses import compute_total_loss
from src.utils.io import ensure_dir, load_yaml, write_csv


USER_CFG = {
    # -----------------------------------------------------------------
    # Core paths: set only these three to train.      
    # -----------------------------------------------------------------
    "train_obdata_dir": "dataset/v1/train/obData_s04",
    "train_field_truth_dir": "dataset/v1/train/fields",
    "train_output_dir": "outputs/train/exp_v1_s04",  # set your output dir here
    # -----------------------------------------------------------------
    # Auto data-cache build (internal; user usually doesn't care).
    # -----------------------------------------------------------------
    "auto_build_dataset_cache": True,
    "dataset_cache_dir": None,  # None -> <train_output_dir>/cache/processed
    "field_kind": "auto",  # auto | lnk | k
    "grid_size": 50,
    "num_steps": 51,
    "val_ratio": 0.1,
    "split_seed": 42,
    # -----------------------------------------------------------------
    # Model/training config.
    # -----------------------------------------------------------------
    "config_path": "configs/default.yaml",
    "experiment_name": "exp_v1_s04",  # for logging and checkpoint metadata
    "device": None,  # None -> use YAML default     
    "seed_override": None,  # None -> use YAML default
    "epochs": None,  # None -> use YAML default
    "batch_size": None,  # None -> use YAML default
    "lr": None,  # None -> use YAML default
    "num_workers": None,  # None -> use YAML default
}


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)


def resolve_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def aggregate_loss(loss_sum: dict[str, float], n: int) -> dict[str, float]:
    return {k: v / max(n, 1) for k, v in loss_sum.items()}


def run_epoch(
    model: nn.Module,
    diffusion: DiffusionScheduler,
    loader: DataLoader,
    sensor_xy: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg_dropout: float,
    loss_weights: dict,
    amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = {"total": 0.0, "lv": 0.0, "lx0": 0.0, "lfft": 0.0, "ltv": 0.0}
    step_count = 0

    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else torch.cpu.amp.autocast

    for batch in loader:
        obs = batch["obs_feat"].to(device)
        x0 = batch["lnk"].to(device)
        bsz = x0.size(0)

        if is_train and cfg_dropout > 0.0:
            # Classifier-free training: randomly zero condition per sample.
            keep = (torch.rand((bsz, 1, 1, 1), device=device) > cfg_dropout).float()
            obs_in = obs * keep
        else:
            obs_in = obs

        t = diffusion.sample_timesteps(bsz, device)
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0, t, noise)
        v_tgt = diffusion.v_target(x0, noise, t)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast_ctx(enabled=amp):
            v_pred = model(x_t, t, obs_in, sensor_xy)
            x0_pred = diffusion.predict_x0_from_v(x_t, t, v_pred)
            total, loss_dict = compute_total_loss(v_pred, v_tgt, x0_pred, x0, loss_weights)

        if is_train:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        loss_sum["total"] += float(total.detach().item())
        for k, v in loss_dict.items():
            loss_sum[k] += float(v.detach().item())
        step_count += 1

    return aggregate_loss(loss_sum, step_count)


def main() -> None:
    ucfg = dict(USER_CFG)
    cfg = load_yaml(ucfg["config_path"])
    ab_cfg = cfg.get("model", {}).get("ablation", {})

    device_str = str(ucfg["device"] or cfg["project"]["device"])
    device = resolve_device(device_str)
    seed = int(ucfg["seed_override"] if ucfg["seed_override"] is not None else cfg["project"]["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_output_dir = Path(ucfg["train_output_dir"])
    ckpt_dir = ensure_dir(train_output_dir / "checkpoints")
    log_dir = ensure_dir(train_output_dir / "logs")
    log_csv = log_dir / "train_log.csv"

    if bool(ucfg["auto_build_dataset_cache"]):
        cache_dir = Path(ucfg["dataset_cache_dir"] or (train_output_dir / "cache" / "processed"))
        built = build_train_val_cache(
            train_ob_dir=ucfg["train_obdata_dir"],
            train_field_dir=ucfg["train_field_truth_dir"],
            out_dir=cache_dir,
            grid_size=int(ucfg["grid_size"]),
            num_steps=int(ucfg["num_steps"]),
            val_ratio=float(ucfg["val_ratio"]),
            seed=int(ucfg["split_seed"]),
            field_kind=str(ucfg["field_kind"]),
        )
        train_h5 = built["train_h5"]
        val_h5 = built["val_h5"]
        stats_json = built["stats_json"]
    else:
        raise ValueError("auto_build_dataset_cache=False is not supported in this streamlined workflow.")

    train_ds = SensorDataset(train_h5, stats_json, normalize=True)
    val_ds = SensorDataset(val_h5, stats_json, normalize=True)

    batch_size = int(ucfg["batch_size"] or cfg["training"]["batch_size"])
    num_workers = int(ucfg["num_workers"] if ucfg["num_workers"] is not None else cfg["training"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    model = build_model_from_config(cfg).to(device)
    ema = EMA(model, decay=float(cfg["training"]["ema_decay"]))
    diffusion = DiffusionScheduler(timesteps=int(cfg["diffusion"]["timesteps"])).to(device)
    sensor_xy = torch.from_numpy(train_ds.sensor_xy_norm).float().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(ucfg["lr"] or cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    amp = bool(cfg["training"]["amp"]) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    epochs = int(ucfg["epochs"] or cfg["training"]["epochs"])
    loss_weights = cfg["loss"]
    cfg_dropout = float(cfg["diffusion"]["cfg_dropout"])
    grad_clip = float(cfg["training"]["grad_clip"])
    experiment_name = str(ucfg.get("experiment_name", "manual_experiment"))

    best_val = float("inf")
    rows = []

    print(
        f"[Config] experiment={experiment_name} "
        f"train_obdata_dir={ucfg['train_obdata_dir']} train_field_truth_dir={ucfg['train_field_truth_dir']} "
        f"train_output_dir={train_output_dir}"
    )
    print(
        "[Ablation] "
        f"disable_token_to_cond_map={bool(ab_cfg.get('disable_token_to_cond_map', False))} "
        f"global_broadcast_channels={int(ab_cfg.get('global_broadcast_channels', 8))} "
        f"disable_global_to_adagn={bool(ab_cfg.get('disable_global_to_adagn', False))} "
        f"disable_sensor_mid_cross_attn={bool(ab_cfg.get('disable_sensor_mid_cross_attn', False))}"
    )
    print(f"[AutoCache] train_h5={train_h5} val_h5={val_h5} stats_json={stats_json}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = run_epoch(
            model=model,
            diffusion=diffusion,
            loader=train_loader,
            sensor_xy=sensor_xy,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg_dropout=cfg_dropout,
            loss_weights=loss_weights,
            amp=amp,
            grad_clip=grad_clip,
        )
        ema.update(model)

        with torch.no_grad():
            va = run_epoch(
                model=ema.ema_model,
                diffusion=diffusion,
                loader=val_loader,
                sensor_xy=sensor_xy,
                optimizer=None,
                scaler=scaler,
                device=device,
                cfg_dropout=0.0,
                loss_weights=loss_weights,
                amp=amp,
                grad_clip=grad_clip,
            )

        dt = time.time() - t0
        row = {
            "epoch": epoch,
            "train_total": tr["total"],
            "train_lv": tr["lv"],
            "train_lx0": tr["lx0"],
            "train_lfft": tr["lfft"],
            "train_ltv": tr["ltv"],
            "val_total": va["total"],
            "val_lv": va["lv"],
            "val_lx0": va["lx0"],
            "val_lfft": va["lfft"],
            "val_ltv": va["ltv"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": dt,
        }
        rows.append(row)
        write_csv(log_csv, fieldnames=row.keys(), rows=rows)

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema.ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "experiment": experiment_name,
            "stats": train_ds.stats,
            "train_obdata_dir": ucfg["train_obdata_dir"],
            "train_field_truth_dir": ucfg["train_field_truth_dir"],
        }
        torch.save(ckpt_state, ckpt_dir / "last.pt")
        if va["total"] < best_val:
            best_val = va["total"]
            torch.save(ckpt_state, ckpt_dir / "best.pt")

        print(
            f"[Epoch {epoch:03d}] "
            f"train={tr['total']:.5f} val={va['total']:.5f} "
            f"(lv={tr['lv']:.4f}, lx0={tr['lx0']:.4f}, lfft={tr['lfft']:.4f}, ltv={tr['ltv']:.4f}) "
            f"time={dt:.1f}s"
        )

    print(f"[OK] checkpoints: {ckpt_dir}")
    print(f"[OK] train log: {log_csv}")


if __name__ == "__main__":
    main()
