from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import csv
from pathlib import Path

import matplotlib
import numpy as np

from src.utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    "log_csv": "outputs/train/exp_v1_baseline/logs/train_log.csv",
    "out_path": None,  # None -> outputs/viz/<log_stem>_loss_curve.png
    "x_key": "epoch",
    "train_key": "train_total",
    "val_key": "val_total",
    "title": "Training Loss Curve",
    "dpi": 200,
}


def load_log_csv(csv_path: str | Path) -> dict[str, np.ndarray]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"train log not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"train log is empty: {csv_path}")

    data: dict[str, np.ndarray] = {}
    for key in reader.fieldnames or []:
        values = []
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                values.append(np.nan)
        data[key] = np.asarray(values, dtype=np.float64)
    return data


def resolve_out_path(log_csv: str | Path, out_path: str | Path | None) -> Path:
    if out_path is not None:
        return Path(out_path)
    log_csv = Path(log_csv)
    return Path("outputs/viz") / f"{log_csv.stem}_loss_curve.png"


def main() -> None:
    cfg = dict(USER_CFG)
    log_csv = Path(cfg["log_csv"])
    out_path = resolve_out_path(log_csv, cfg.get("out_path"))
    x_key = str(cfg.get("x_key", "epoch"))
    train_key = str(cfg.get("train_key", "train_total"))
    val_key = str(cfg.get("val_key", "val_total"))
    title = str(cfg.get("title", "Training Loss Curve"))
    dpi = int(cfg.get("dpi", 200))

    data = load_log_csv(log_csv)
    for key in (x_key, train_key, val_key):
        if key not in data:
            raise KeyError(f"Column not found in {log_csv}: {key}")

    x = data[x_key]
    train_loss = data[train_key]
    val_loss = data[val_key]

    ensure_dir(out_path.parent)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.0), constrained_layout=True)
    ax.plot(x, train_loss, color="#1f77b4", linewidth=1.8, label="train_loss")
    ax.plot(x, val_loss, color="#d62728", linewidth=1.8, label="val_loss")
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel("loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[OK] loss curve -> {out_path}")


if __name__ == "__main__":
    main()
