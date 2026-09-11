"""Fits the small DN-pool -> command readout with a cross-entropy-method
(CEM) evolutionary search against a flight-stabilization objective (make
forward progress down the corridor, don't hit anything). The connectome
weights are never touched here -- only this ~30-parameter readout is
optimized, exactly as described in the project plan (real/literature-derived
biology stays fixed; only the final decoding layer is learned).

Defaults are deliberately small-scale (a handful of iterations/candidates,
short rollouts, a reduced column patch) so this finishes in well under a
minute on a laptop -- it's meant to demonstrate the training loop working
end-to-end, not to be a tuned, exhaustively-optimized controller.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..connectome.constants import READOUT_INPUT_TYPES
from ..sim.run import EpisodeResult, build_graph, run_episode
from .readout import N_OUTPUTS, Readout


def _score(result: EpisodeResult) -> float:
    collision_penalty = 6.0 if result.collided else 0.0
    altitude_penalty = 2.0 * result.mean_altitude_error
    return (
        result.forward_progress
        + 1.5 * min(result.min_clearance, 1.0)
        - collision_penalty
        - altitude_penalty
    )


def train_readout(
    n_iterations: int = 10,
    population: int = 10,
    elite_frac: float = 0.3,
    sigma0: float = 0.4,
    n_columns: int = 60,
    n_steps: int = 150,
    dt: float = 0.005,
    seeds: Sequence[int] = (0, 1),
    rng_seed: int = 0,
    verbose: bool = True,
) -> Readout:
    graph = build_graph(n_columns=n_columns)
    dim = N_OUTPUTS * len(READOUT_INPUT_TYPES) + N_OUTPUTS
    rng = np.random.default_rng(rng_seed)

    mean = np.zeros(dim)
    std = np.full(dim, sigma0)
    n_elite = max(1, int(round(population * elite_frac)))

    best_flat, best_score = mean.copy(), -np.inf

    for iteration in range(n_iterations):
        candidates = mean + std * rng.standard_normal((population, dim))
        scores = np.empty(population)

        for i, flat in enumerate(candidates):
            readout = Readout.from_flat(flat)
            episode_scores = [
                _score(run_episode(graph, readout, n_steps=n_steps, dt=dt, seed=seed))
                for seed in seeds
            ]
            scores[i] = float(np.mean(episode_scores))
            if scores[i] > best_score:
                best_score, best_flat = scores[i], flat.copy()

        elite = candidates[np.argsort(-scores)[:n_elite]]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-2

        if verbose:
            print(
                f"[train_readout] iter {iteration:02d}  "
                f"mean_score={scores.mean():+.3f}  best_score={best_score:+.3f}"
            )

    return Readout.from_flat(best_flat)
