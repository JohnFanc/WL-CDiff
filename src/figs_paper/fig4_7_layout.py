from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path as _Path

    _project_root = _Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.data.sensor_utils import load_sensor_config
from src.utils.io import ensure_dir
from src.figs_paper import style

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    # 基本网格尺寸（当前研究区为 50x50 网格）
    "grid_size": 50,
    # 输出图片路径
    "out_path": "outputs/viz/table_points_8_S36_p.png",
    # Monitoring well 坐标表路径
    "sensors_csv": "configs/sensors_36.csv",
    # 是否显示 pumping well 的编号
    "show_point_id": True,
    # 是否显示网格线（期刊版：开启浅色辅助网格，便于读坐标）
    "show_grid_lines": True,
    # 是否显示图例
    "show_legend": True,
    # 是否显示 pumping well（绿色圆点）
    "show_pumping_well": True,
    # 是否显示 monitoring well（黄色星号）
    "show_monitoring_well": True,
    # pumping well 点大小
    "marker_size": 72.0,
    # monitoring well 点大小
    "monitoring_marker_size": 100.0,
    # 研究区背景颜色（期刊版：调浅，降低油墨量）
    "study_area_color": "#9DBDE8",
    # pumping well 颜色与边框颜色
    "point_color": "#2E8B57",
    "point_edge_color": "white",
    "monitoring_color": "#FFD700",
    # monitoring well 颜色与边框颜色
    "monitoring_edge_color": "black",
    # pumping well 编号可尝试放置的候选偏移位置
    "point_label_candidates": [
        (0.9, -0.9),
        (0.9, 0.9),
        (-0.9, -0.9),
        (-0.9, 0.9),
        (1.25, 0.0),
        (-1.25, 0.0),
        (0.0, -1.25),
        (0.0, 1.25),
    ],
    # 编号与任一点的最小避让距离
    "point_label_margin": 1.05,
    # 手动覆盖某些点编号的位置：{点编号: (dx, dy)}
    "point_label_overrides": {
        6: (1.0, 0.9),
    },
    # Pumping well 列表，格式为 {"id": 编号, "x": 横坐标, "y": 纵坐标}
    "points": [
        {"id": 1, "x": 9, "y": 9},
        {"id": 2, "x": 39, "y": 9},
        {"id": 3, "x": 19, "y": 19},
        {"id": 4, "x": 29, "y": 19},
        {"id": 5, "x": 19, "y": 29},
        {"id": 6, "x": 29, "y": 29},
        {"id": 7, "x": 9, "y": 39},
        {"id": 8, "x": 39, "y": 39},
    ],
}


def _choose_label_positions(
    point_xy: np.ndarray,
    monitoring_xy: np.ndarray,
    grid_size: int,
    candidates: list[tuple[float, float]],
    margin: float,
) -> np.ndarray:
    all_points = np.vstack([point_xy, monitoring_xy]).astype(np.float32)
    label_pos = np.zeros((point_xy.shape[0], 2), dtype=np.float32)

    for i, (x, y) in enumerate(point_xy.astype(np.float32)):
        best_pos = np.asarray([x + candidates[0][0], y + candidates[0][1]], dtype=np.float32)
        best_score = -1e9
        for dx, dy in candidates:
            px = x + dx
            py = y + dy
            if not (-0.2 <= px <= grid_size - 0.8 and -0.2 <= py <= grid_size - 0.8):
                continue
            pos = np.asarray([px, py], dtype=np.float32)
            dists = np.linalg.norm(all_points - pos[None, :], axis=1)
            min_dist = float(dists.min())
            # Prefer placements that stay away from all points, especially within the margin radius.
            score = min_dist
            if min_dist < margin:
                score -= 10.0
            if score > best_score:
                best_score = score
                best_pos = pos
        label_pos[i] = best_pos
    return label_pos


def _legend_handles(cfg: dict) -> list:
    handles = [Patch(facecolor=cfg["study_area_color"], edgecolor="black", label="Study area")]
    if cfg.get("show_pumping_well", True):
        handles.append(Line2D([0], [0], marker="o", linestyle="None",
                              markerfacecolor=cfg["point_color"], markeredgecolor=cfg["point_edge_color"],
                              color="none", markersize=7.5, label="Pumping well"))
    if cfg.get("show_monitoring_well", True):
        handles.append(Line2D([0], [0], marker="*", linestyle="None",
                              markerfacecolor=cfg["monitoring_color"], markeredgecolor=cfg["monitoring_edge_color"],
                              color="none", markersize=7.0, label="Monitoring well"))
    return handles


def _draw_panel(ax, sensors_csv: str, *, show_ylabel: bool = True,
                show_left_noflow: bool = True, show_right_noflow: bool = True) -> list:
    """Draw the study-area layout (wells + boundaries + flow arrow) onto one axes.
    Returns the legend handles (legend itself is added by the caller)."""
    cfg = USER_CFG
    grid_size = int(cfg["grid_size"])
    study_area_color = str(cfg["study_area_color"])
    point_color = str(cfg["point_color"])
    point_edge_color = str(cfg["point_edge_color"])
    monitoring_color = str(cfg["monitoring_color"])
    monitoring_edge_color = str(cfg["monitoring_edge_color"])
    marker_size = float(cfg.get("marker_size", 72.0))
    monitoring_marker_size = float(cfg.get("monitoring_marker_size", 100.0))
    show_point_id = bool(cfg.get("show_point_id", True))
    show_pumping_well = bool(cfg.get("show_pumping_well", True))
    show_monitoring_well = bool(cfg.get("show_monitoring_well", True))
    show_grid_lines = bool(cfg.get("show_grid_lines", True))
    candidates = [tuple(map(float, t)) for t in cfg.get("point_label_candidates", [(0.9, -0.9)])]
    margin = float(cfg.get("point_label_margin", 1.05))
    overrides = {int(k): tuple(map(float, v)) for k, v in dict(cfg.get("point_label_overrides", {})).items()}

    pts = cfg["points"]
    point_ids = np.asarray([int(p["id"]) for p in pts], dtype=np.int32)
    point_xy = np.asarray([[int(p["x"]), int(p["y"])] for p in pts], dtype=np.int32)
    monitoring_xy = load_sensor_config(sensors_csv)
    visible = monitoring_xy if show_monitoring_well else np.zeros((0, 2), dtype=np.int32)
    label_xy = _choose_label_positions(point_xy, visible, grid_size, candidates, margin)
    for i, pid in enumerate(point_ids):
        if int(pid) in overrides:
            dx, dy = overrides[int(pid)]
            label_xy[i] = np.asarray([point_xy[i, 0] + dx, point_xy[i, 1] + dy], dtype=np.float32)

    ax.set_facecolor(study_area_color)
    if show_monitoring_well:
        ax.scatter(monitoring_xy[:, 0], monitoring_xy[:, 1], s=monitoring_marker_size,
                   c=monitoring_color, marker="*", edgecolors=monitoring_edge_color,
                   linewidths=0.9, zorder=2)
    if show_pumping_well:
        ax.scatter(point_xy[:, 0], point_xy[:, 1], s=marker_size, c=point_color,
                   edgecolors=point_edge_color, linewidths=0.9, zorder=3)
    if show_point_id and show_pumping_well:
        for pid, (lx, ly) in zip(point_ids, label_xy):
            ax.text(lx, ly, str(pid), fontsize=style.pt(8), color="white", weight="bold",
                    ha="center", va="center", zorder=4)

    # 自上而下水流方向箭头（右侧空白列内，避开井位）
    ax.annotate("", xy=(47.8, 42.0), xytext=(47.8, 7.0),
                arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 2.2, "alpha": 0.85}, zorder=5)
    ax.text(45.9, 24.5, "Groundwater flow", rotation=270, ha="center", va="center",
            fontsize=style.pt(9), fontweight="bold", color="black", zorder=5)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)
    ax.set_aspect("equal")
    style.style_field_axes_meters(ax, grid_size)
    if not show_ylabel:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.grid(show_grid_lines, linestyle="--", linewidth=0.5, alpha=0.35, color="0.6")

    ax.text(0.5, 1.04, "H=105m", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=style.pt(11), fontweight="bold", clip_on=False)
    ax.text(0.5, -0.125, "H=100m", transform=ax.transAxes, ha="center", va="top",
            fontsize=style.pt(11), fontweight="bold", clip_on=False)
    if show_left_noflow:
        ax.text(-0.225, 0.5, "No-flow boundary", transform=ax.transAxes, ha="center", va="center",
                rotation=90, fontsize=style.pt(10.5), fontweight="bold", clip_on=False)
    if show_right_noflow:
        ax.text(1.13, 0.5, "No-flow boundary", transform=ax.transAxes, ha="center", va="center",
                rotation=270, fontsize=style.pt(10.5), fontweight="bold", clip_on=False)

    return _legend_handles(cfg)


def main() -> None:
    """Single-panel layout (Fig.4: study area with monitoring wells)."""
    style.apply_style()
    out_path = Path(USER_CFG["out_path"])
    ensure_dir(out_path.parent)
    # Keep the study-area panel compact while reserving a narrow band above the
    # axes for the legend.  This removes the large right-hand blank area in the
    # previous layout and keeps the legend visually associated with the map.
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.9), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.84, bottom=0.14)
    handles = _draw_panel(ax, str(USER_CFG["sensors_csv"]))

    # Move the axis titles closer to their axes, as requested in the review.
    ax.xaxis.labelpad = 1.0
    ax.yaxis.labelpad = 1.0

    if bool(USER_CFG.get("show_legend", True)):
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.00, 1.015),
                  frameon=True, facecolor="white", edgecolor="black", fancybox=False,
                  framealpha=1.0, fontsize=style.pt(10), borderpad=0.6,
                  handlelength=1.2, borderaxespad=0.0)
    fig.savefig(out_path, dpi=style.DPI)
    plt.close(fig)
    print(f"[OK] point location figure -> {out_path}")


def render_combined(sensor_csvs: list[str], titles: list[str], out_path: str | Path) -> None:
    """Two-panel layout sharing a single legend (Fig.7: 18-well and 4-well layouts)."""
    style.apply_style()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 7.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.80, bottom=0.10, wspace=0.62)
    handles = None
    for i, (ax, csv, title) in enumerate(zip(axes, sensor_csvs, titles)):
        # each panel is a full, self-contained study area -> its own y-axis and
        # both left/right no-flow boundaries.
        handles = _draw_panel(ax, csv, show_ylabel=True,
                              show_left_noflow=True, show_right_noflow=True)
        ax.set_title(title, fontsize=style.pt(13), fontweight="bold", pad=30)
    # single shared legend, same style as Fig.4 (vertical, white box, top-right)
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.0),
               frameon=True, facecolor="white", edgecolor="black", fancybox=False,
               framealpha=1.0, fontsize=style.pt(10), borderpad=0.8, handlelength=1.2)
    fig.savefig(out_path, dpi=style.DPI)
    plt.close(fig)
    print(f"[OK] combined layout figure -> {out_path}")


if __name__ == "__main__":
    main()
