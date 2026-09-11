#!/usr/bin/env python3
"""Runs the closed loop end-to-end: real connectome-derived spiking network
piloting a simulated drone through a procedural corridor, and saves plots.

Usage:
    python scripts/demo.py [--n-columns 100] [--n-steps 600] [--out output/] [--skip-training]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fly_drone_pilot.model.readout import Readout  # noqa: E402
from fly_drone_pilot.model.train_readout import train_readout  # noqa: E402
from fly_drone_pilot.sim.environment import Environment  # noqa: E402
from fly_drone_pilot.sim.run import build_graph, run_episode  # noqa: E402
from fly_drone_pilot.viz.plots import save_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-columns", type=int, default=100, help="real hex columns to model")
    parser.add_argument("--n-steps", type=int, default=600, help="simulated timesteps to fly")
    parser.add_argument("--dt", type=float, default=0.005, help="simulation timestep, seconds")
    parser.add_argument("--seed", type=int, default=0, help="environment/episode seed")
    parser.add_argument("--out", type=str, default="output", help="output directory for plots")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="use an untrained (zero-weight) readout instead of training one",
    )
    args = parser.parse_args()

    print(f"Building connectome-derived instance graph ({args.n_columns} real hex columns)...")
    graph = build_graph(n_columns=args.n_columns)
    print(f"  {graph.n_units} modeled units "
          f"({graph.weight_fast.nnz + graph.weight_delayed.nnz} synaptic edges)")

    if args.skip_training:
        readout = Readout()
        print("Using untrained (zero-weight) readout.")
    else:
        print("Training the DN-pool -> command readout (connectome weights stay fixed)...")
        readout = train_readout(verbose=True)

    print(f"Flying {args.n_steps} steps ({args.n_steps * args.dt:.1f} simulated seconds)...")
    result = run_episode(graph, readout, n_steps=args.n_steps, dt=args.dt, seed=args.seed, record=True)

    print(f"forward progress: {result.forward_progress:.2f} m")
    print(f"collided: {result.collided}  (survived {result.n_steps_survived}/{args.n_steps} steps)")
    print(f"min clearance: {result.min_clearance:.2f} m")

    environment = Environment(seed=args.seed)
    save_all(result, graph, environment, args.dt, args.out)
    print(f"Saved plots to {args.out}/")


if __name__ == "__main__":
    main()
