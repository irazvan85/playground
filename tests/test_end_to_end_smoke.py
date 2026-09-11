import numpy as np

from fly_drone_pilot.model.readout import Readout
from fly_drone_pilot.sim.run import build_graph, run_episode


def test_closed_loop_runs_and_stays_finite():
    graph = build_graph(n_columns=50)
    readout = Readout()
    result = run_episode(graph, readout, n_steps=150, dt=0.005, seed=0, record=True)

    assert np.isfinite(result.trajectory).all()
    assert np.isfinite(result.dn_rate_history).all()
    assert np.isfinite(result.forward_progress)
    assert result.spike_raster.dtype == bool


def test_trained_readout_beats_zero_readout_on_average():
    from fly_drone_pilot.model.train_readout import train_readout

    graph = build_graph(n_columns=60)
    zero_readout = Readout()
    trained = train_readout(
        n_iterations=3, population=4, n_columns=40, n_steps=60, seeds=(0,), verbose=False
    )

    def score(readout):
        results = [run_episode(graph, readout, n_steps=200, seed=s) for s in (0, 1, 2)]
        return np.mean(
            [r.forward_progress - (6.0 if r.collided else 0.0) for r in results]
        )

    assert score(trained) >= score(zero_readout)
