from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from pathlib import Path

import matplotlib

from src.data.sensor_utils import read_head_file
from src.utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    # Path to one Head*.h5 file.
    "head_file": "dataset/v1/train/heads/Head0001.h5",
    # If head_file is None, fallback to the first Head*.h5 under this directory.
    "heads_dir": "dataset/v1/train/heads",
    # Water-level period to visualize.
    "period": 1,
    # True: period is 1-based (1..T). False: 0-based (0..T-1).
    "use_one_based_period": True,
    "grid_size": 50,
    "out_path": None,  # None -> outputs/viz/<head_file_stem>_period_xx.png
    "cmap": "turbo",
    "vmin": None,
    "vmax": None,
}


def resolve_head_file(head_file: str | None, heads_dir: str | Path) -> Path:
    if head_file:
        path = Path(head_file)
        if not path.exists():
            raise FileNotFoundError(f"Head file not found: {path}")
        return path

    heads_dir = Path(heads_dir)
    candidates = sorted(heads_dir.glob("Head*.h5"))
    if not candidates:
        raise FileNotFoundError(f"No Head*.h5 found in: {heads_dir}")
    return candidates[0]


def resolve_step_index(period: int, one_based: bool, num_steps: int) -> int:
    idx = period - 1 if one_based else period
    if idx < 0 or idx >= num_steps:
        if one_based:
            raise ValueError(f"period out of range: {period}. Expected 1..{num_steps}.")
        raise ValueError(f"period out of range: {period}. Expected 0..{num_steps - 1}.")
    return idx


def main() -> None:
    grid_size = int(USER_CFG["grid_size"])
    head_file = resolve_head_file(USER_CFG.get("head_file"), USER_CFG["heads_dir"])
    period = int(USER_CFG.get("period", 1))
    one_based = bool(USER_CFG.get("use_one_based_period", True))
    cmap = str(USER_CFG.get("cmap", "turbo"))
    vmin = USER_CFG.get("vmin")
    vmax = USER_CFG.get("vmax")

    head_3d, times = read_head_file(head_file, grid_size=grid_size)
    step_idx = resolve_step_index(period, one_based, head_3d.shape[0])
    head_map = head_3d[step_idx]

    out_path = USER_CFG.get("out_path")
    if out_path is None:
        out_path = f"outputs/viz/{head_file.stem}_period_{period:02d}.png"
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.0), constrained_layout=True)
    im = ax.imshow(head_map, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"Head Field | {head_file.name} | period={period} (idx={step_idx}) | t={float(times[step_idx]):.4g}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Head")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] head field figure -> {out_path}")


if __name__ == "__main__":
    main()
