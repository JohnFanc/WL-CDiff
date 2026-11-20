from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from pathlib import Path

import matplotlib

from src.data.sensor_utils import load_sensor_config, read_lnk_file
from src.utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    "sensors_csv": "configs/sensors_18.csv",
    "lnk_field_path": "dataset/v1/train/fields/lnKfield_0007.csv",
    "grid_size": 50,
    "out_path": "outputs/viz/sensor_locations_18.png",
    "show_sensor_id": True,
    "marker_size": 60.0,
    "field_cmap": "viridis",
    "sensor_color": "#E63946",
    "sensor_edge_color": "white",
}


def main() -> None:
    sensors_csv = str(USER_CFG["sensors_csv"])
    lnk_field_path = str(USER_CFG["lnk_field_path"])
    grid_size = int(USER_CFG["grid_size"])
    out_path = Path(USER_CFG["out_path"])
    show_sensor_id = bool(USER_CFG.get("show_sensor_id", True))
    marker_size = float(USER_CFG.get("marker_size", 60.0))
    field_cmap = str(USER_CFG.get("field_cmap", "viridis"))
    sensor_color = str(USER_CFG.get("sensor_color", "#E63946"))
    sensor_edge_color = str(USER_CFG.get("sensor_edge_color", "white"))

    sensor_xy = load_sensor_config(sensors_csv)  # (S,2)
    lnk_field = read_lnk_file(lnk_field_path, grid_size=grid_size)
    ensure_dir(out_path.parent)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.0), constrained_layout=True)
    im = ax.imshow(lnk_field, cmap=field_cmap, origin="upper")
    ax.scatter(
        sensor_xy[:, 0],
        sensor_xy[:, 1],
        s=marker_size,
        c=sensor_color,
        edgecolors=sensor_edge_color,
        linewidths=0.9,
    )

    if show_sensor_id:
        for i, (x, y) in enumerate(sensor_xy):
            ax.text(
                x + 0.45,
                y + 0.35,
                str(i),
                fontsize=8,
                color="white",
                weight="bold",
            )

    ax.set_title(f"Observation Well Locations ({sensor_xy.shape[0]} points)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ln K")

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] sensor location figure -> {out_path}")


if __name__ == "__main__":
    main()
