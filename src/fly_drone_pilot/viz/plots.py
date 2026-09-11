"""Headless (Agg-backend) plots for the demo: no display is assumed."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..connectome.constants import DN_TYPES
from ..connectome.graph import InstanceGraph
from ..sim.run import EpisodeResult


def plot_trajectory(result: EpisodeResult, environment, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    traj = result.trajectory

    x_margin = max(1.0, 0.15 * (traj[:, 0].max() - traj[:, 0].min()))
    x_lo, x_hi = traj[:, 0].min() - x_margin, traj[:, 0].max() + x_margin

    ax = axes[0]
    ax.plot(traj[:, 0], traj[:, 1], color="tab:blue", lw=1.5)
    ax.scatter(traj[0, 0], traj[0, 1], color="green", zorder=3, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], color="red" if result.collided else "black", zorder=3, label="end")
    for cx, cy in environment.pillars:
        if x_lo <= cx <= x_hi:
            ax.add_patch(plt.Circle((cx, cy), environment.pillar_radius, color="gray", alpha=0.6))
    ax.axhline(environment.half_width, color="k", lw=0.5)
    ax.axhline(-environment.half_width, color="k", lw=0.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-environment.half_width * 1.1, environment.half_width * 1.1)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (lateral, m)")
    ax.set_title("Top-down trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.plot(traj[:, 0], traj[:, 2], color="tab:orange", lw=1.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(environment.ceiling, color="k", lw=0.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.2, environment.ceiling + 0.2)
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("z (altitude, m)")
    ax.set_title("Side-view altitude")

    fig.suptitle(
        f"forward progress={result.forward_progress:.2f} m  "
        f"collided={result.collided}  min_clearance={result.min_clearance:.2f} m"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spike_raster(result: EpisodeResult, graph: InstanceGraph, dt: float, out_path: str, max_units: int = 400) -> None:
    raster = result.spike_raster
    n_units = raster.shape[1]
    unit_idx = np.linspace(0, n_units - 1, min(max_units, n_units)).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    times, units = np.nonzero(raster[:, unit_idx])
    ax.scatter(times * dt, units, s=1, color="black")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("unit index (subsample)")
    ax.set_title("Connectome-driven spike raster (real-cell-type units)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dn_rates(result: EpisodeResult, dt: float, out_path: str) -> None:
    history = result.dn_rate_history
    t = np.arange(history.shape[0]) * dt

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(DN_TYPES):
        ax.plot(t, history[:, i], label=name, lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("Descending-neuron pool firing rates (drive readout -> drone commands)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eye_snapshot(result: EpisodeResult, graph: InstanceGraph, out_path: str) -> None:
    if result.luminance_snapshot is None:
        return
    xs, ys, vals = [], [], []
    for column_id, luminance in result.luminance_snapshot.items():
        x, y = graph.column_xy[column_id]
        xs.append(x)
        ys.append(y)
        vals.append(luminance)

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(xs, ys, c=vals, cmap="gray", vmin=0, vmax=1, s=40)
    ax.set_xlabel("retinotopic x (real hex-column coordinate)")
    ax.set_ylabel("retinotopic y (real hex-column coordinate)")
    ax.set_title("Compound-eye luminance snapshot\n(real column layout, mid-flight)")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="luminance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all(result: EpisodeResult, graph: InstanceGraph, environment, dt: float, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_trajectory(result, environment, str(out / "trajectory.png"))
    plot_spike_raster(result, graph, dt, str(out / "spike_raster.png"))
    plot_dn_rates(result, dt, str(out / "dn_rates.png"))
    plot_eye_snapshot(result, graph, str(out / "eye_snapshot.png"))
