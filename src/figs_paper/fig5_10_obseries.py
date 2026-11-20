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

from src.data.sensor_utils import parse_sample_id
from src.utils.io import ensure_dir
from src.figs_paper import style

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USER_CFG = {
    # Priority 1: direct sequence file path.
    # Example: "dataset/v1/train/obData/Head0001.npz"
    "ob_file": "dataset/v1/test/obData_s04/Head1501.npz",
    # Priority 2+: choose a file from this directory when ob_file is None.
    "ob_dir": None,
    # Optional selectors under ob_dir.
    "sequence_name": None,  # Example: "Head0001.npz"
    "sample_id": None,  # Example: 1
    # Plot options.
    "sensor_ids": None,  # None -> all sensors; or [0, 1, 2]
    "feature": "h_abs",  # h_abs | delta | dhdt
    "out_path": None,  # None -> outputs/viz/<file>_<feature>.png
    # --- paper-figure additions ---
    "xlabel": "Time (d)",
    "ylabel": "Head (m)",            # override code-style "h_abs"
    "pumping_switch_day": 11.0,      # scheme A spans days 1-10, scheme B starts on day 11
    "inset_clean_file": None,        # if set, draw a clean-vs-noisy zoom inset (used by fig.10)
    "inset_window": (12.0, 20.0),    # day window for the zoom inset
}


def _resolve_ob_dir(ob_dir: str | Path) -> Path:
    ob_dir = Path(ob_dir)
    if not ob_dir.exists():
        raise FileNotFoundError(f"obData dir not found: {ob_dir}. Run src.data.extract_head_obs first.")
    return ob_dir


def _resolve_ob_file(plot_cfg: dict) -> Path:
    if plot_cfg.get("ob_file"):
        p = Path(str(plot_cfg["ob_file"]))
        if not p.exists():
            raise FileNotFoundError(f"ob file not found: {p}")
        return p

    ob_dir = _resolve_ob_dir(str(plot_cfg["ob_dir"]))
    files = sorted(ob_dir.glob("Head*.npz"))
    if not files:
        raise ValueError(f"No .npz sequence files in {ob_dir}. Run src.data.extract_head_obs first.")

    if plot_cfg.get("sequence_name"):
        target = ob_dir / str(plot_cfg["sequence_name"])
        if not target.exists():
            raise FileNotFoundError(f"sequence file not found in obData: {target}")
        return target

    if plot_cfg.get("sample_id") is not None:
        sid = int(plot_cfg["sample_id"])
        matched = [fp for fp in files if parse_sample_id(fp) == sid]
        if not matched:
            raise FileNotFoundError(f"No sequence file with sample_id={sid} in {ob_dir}")
        return matched[0]

    return files[0]


def _resolve_sensor_ids(sensor_ids: object, num_sensors: int) -> list[int]:
    if sensor_ids is None:
        return list(range(num_sensors))
    if isinstance(sensor_ids, int):
        sensor_ids = [sensor_ids]
    ids = [int(x) for x in sensor_ids]
    for i in ids:
        if i < 0 or i >= num_sensors:
            raise ValueError(f"sensor id {i} out of range [0, {num_sensors - 1}]")
    return ids


def _select_feature(obs_abs: np.ndarray, obs_feat: np.ndarray, feature: str) -> tuple[np.ndarray, str]:
    feature = feature.lower()
    if feature == "h_abs":
        return obs_abs, "h_abs"
    if feature == "delta":
        return obs_feat[:, :, 1], "h_abs - h_steady"
    if feature == "dhdt":
        return obs_feat[:, :, 2], "dh/dt"
    raise ValueError(f"feature must be one of h_abs|delta|dhdt, got: {feature}")


def main() -> None:
    plot_cfg = dict(USER_CFG)
    ob_file = _resolve_ob_file(plot_cfg)
    data = np.load(ob_file)
    obs_abs = np.asarray(data["obs_abs"], dtype=np.float32)  # (S,T)
    obs_feat = np.asarray(data["obs_feat"], dtype=np.float32)  # (S,T,3)
    times = np.asarray(data["times"], dtype=np.float32)  # (T,)
    sample_id = int(data["sample_id"]) if "sample_id" in data else -1

    series, y_label = _select_feature(obs_abs, obs_feat, str(plot_cfg.get("feature", "h_abs")))
    sensor_ids = _resolve_sensor_ids(plot_cfg.get("sensor_ids"), series.shape[0])

    if plot_cfg.get("out_path"):
        out_path = Path(plot_cfg["out_path"])
    else:
        out_path = Path(f"outputs/viz/{ob_file.stem}_{plot_cfg.get('feature', 'h_abs')}.png")
    ensure_dir(out_path.parent)

    style.apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(9.2, 5.0), constrained_layout=True)
    for sid in sensor_ids:
        ax.plot(times, series[sid], linewidth=0.9, alpha=0.8)

    ax.set_xlabel(str(plot_cfg.get("xlabel") or "time"))
    ax.set_ylabel(str(plot_cfg.get("ylabel") or y_label))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # Timeline: [0, t0] is the steady-state computation period; pumping starts at t0
    # (transient onset, day 1) and switches scheme at `pumping_switch_day` (day 11).
    t0 = float(times.min())
    xtrans = ax.get_xaxis_transform()  # (data x, axes y)
    ax.set_xlim(0.0, float(times.max()))
    # steady-state period band
    ax.axvspan(0.0, t0, color="0.90", zorder=0)
    ax.text(t0 * 0.5, 0.5, "steady state", transform=xtrans,
            ha="center", va="center", rotation=90, fontsize=style.pt(10), color="0.45")
    # pumping start
    ax.axvline(t0, color="0.35", linestyle="--", linewidth=1.1, zorder=1)
    ax.annotate("pumping start", xy=(t0, 0.985), xycoords=xtrans,
                xytext=(4, -2), textcoords="offset points",
                ha="left", va="top", fontsize=style.pt(12), color="0.25")
    # pumping scheme change
    switch_day = plot_cfg.get("pumping_switch_day")
    if switch_day is not None and t0 <= float(switch_day) <= float(times.max()):
        ax.axvline(float(switch_day), color="0.35", linestyle="--", linewidth=1.1, zorder=1)
        ax.annotate("pumping change", xy=(float(switch_day), 0.985), xycoords=xtrans,
                    xytext=(6, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=style.pt(12), color="0.25")

    # Show the event x-coordinates (pumping start / change) explicitly on the time axis.
    events = [t0] + ([float(switch_day)] if switch_day is not None else [])
    ticks = sorted({0.0, 5.0, 15.0, 20.0, *events})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(round(v))}" for v in ticks])
    for lbl, v in zip(ax.get_xticklabels(), ticks):
        if v in events:
            lbl.set_color("0.15"); lbl.set_fontweight("bold")

    # Optional clean-vs-noisy zoom inset (fig.10): make the 0.03 m noise visible.
    inset_clean = plot_cfg.get("inset_clean_file")
    if inset_clean:
        clean = np.asarray(np.load(inset_clean)["obs_abs"], dtype=np.float32)
        sidx = int(np.argmax(np.ptp(series, axis=1)))  # NumPy 2-compatible
        w0, w1 = (float(v) for v in plot_cfg.get("inset_window", (12.0, 20.0)))
        mask = (times >= w0) & (times <= w1)
        axin = ax.inset_axes([0.60, 0.12, 0.37, 0.40])
        axin.plot(times[mask], clean[sidx][mask], color="#1f77b4", lw=1.6, label="Clean")
        axin.plot(times[mask], series[sidx][mask], color="#d62728", lw=0.9, alpha=0.9, label="Noisy")
        axin.set_title(f"Well S{sidx} (zoom)", fontsize=style.pt(8.5))
        axin.tick_params(labelsize=style.pt(7))
        axin.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
        axin.legend(fontsize=style.pt(7.5), loc="upper right", framealpha=0.9)

    fig.savefig(out_path, dpi=style.DPI)
    plt.close(fig)
    print(f"[OK] sequence figure -> {out_path}")


if __name__ == "__main__":
    main()
