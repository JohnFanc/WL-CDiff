from __future__ import annotations

"""
Generate clean square ICONS for the Input column of Figure 1 (model overview):
  icon_series.png     water-level series  (a few representative wells)
  icon_wells.png      monitoring-well locations in the domain
  icon_grid_mesh.png  the model grid (mesh)            <- option A for "grid"
  icon_grid_field.png a representative lnK field        <- option B for "grid"

All icons share one visual language: square, no axis ticks/labels, thin frame.
Run from project root the project root:
  python -m src.figs_paper.fig1_icons
"""

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from pathlib import Path

import matplotlib
import numpy as np

from src.data.sensor_utils import load_sensor_config, read_lnk_file
from src.utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("src/figs_paper/out/icons")
DPI = 300
SIZE = (2.3, 2.3)
FRAME = "0.35"          # thin frame color
LIGHT_BLUE = "#DCE7F6"  # light domain fill (consistent, muted)


def _bare_axes(ax, frame=True):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame)
        s.set_color(FRAME)
        s.set_linewidth(1.0)


def icon_series(out: Path) -> None:
    d = np.load("dataset/v1/test/obData/Head1506.npz")
    obs = np.asarray(d["obs_abs"], dtype=float)   # (S,T)
    t = np.asarray(d["times"], dtype=float)
    order = np.argsort(obs.mean(axis=1))
    pick = order[np.linspace(0, len(order) - 1, 7).astype(int)]   # spread across range
    colors = plt.cm.tab10(np.linspace(0, 1, len(pick)))
    fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
    for c, i in zip(colors, pick):
        ax.plot(t, obs[i], lw=1.6, color=c, alpha=0.95)
    ax.margins(x=0.02, y=0.06)
    _bare_axes(ax)
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("[OK]", out)


def icon_wells(out: Path) -> None:
    xy = load_sensor_config("configs/sensors_36.csv").astype(float)  # grid index
    fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
    ax.set_facecolor(LIGHT_BLUE)
    ax.scatter(xy[:, 0], xy[:, 1], s=70, marker="*",
               c="#FFD700", edgecolors="black", linewidths=0.7, zorder=3)
    ax.set_xlim(-1, 50); ax.set_ylim(50, -1); ax.set_aspect("equal")
    _bare_axes(ax)
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("[OK]", out)


def icon_grid_mesh(out: Path) -> None:
    fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
    ax.set_facecolor(LIGHT_BLUE)
    for k in range(0, 51, 5):                       # mesh lines every 5 cells
        ax.axhline(k, color="white", lw=0.9)
        ax.axvline(k, color="white", lw=0.9)
    ax.set_xlim(0, 50); ax.set_ylim(50, 0); ax.set_aspect("equal")
    _bare_axes(ax)
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("[OK]", out)


def icon_grid_field(out: Path) -> None:
    field = read_lnk_file("dataset/v1/test/fields/lnKfield_1506.csv", grid_size=50)
    fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
    ax.imshow(field, cmap="viridis", origin="upper")
    ax.set_aspect("equal")
    _bare_axes(ax)
    fig.savefig(out, dpi=DPI); plt.close(fig)
    print("[OK]", out)


def main() -> None:
    ensure_dir(OUT)
    icon_series(OUT / "icon_series.png")
    icon_wells(OUT / "icon_wells.png")
    icon_grid_mesh(OUT / "icon_grid_mesh.png")
    icon_grid_field(OUT / "icon_grid_field.png")


if __name__ == "__main__":
    main()
