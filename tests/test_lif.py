import numpy as np

from fly_drone_pilot.connectome.graph import build_instance_graph
from fly_drone_pilot.connectome.loader import BundledLiteratureSource
from fly_drone_pilot.model.lif import LIFNetwork, LIFParams, RateTracker


def _build_graph(n_columns=50):
    return build_instance_graph(BundledLiteratureSource(n_columns=n_columns).load())


def test_no_input_stays_silent_and_finite():
    graph = _build_graph()
    lif = LIFNetwork(graph, LIFParams(dt=0.005))
    for _ in range(100):
        spikes = lif.step(np.zeros(graph.n_units))
    assert not spikes.any()
    assert np.isfinite(lif.v).all()
    assert np.isfinite(lif.activity).all()


def test_driven_lamina_cells_relay_to_medulla():
    """L1 -> Mi1 is a real, direct hemibrain/FlyWire connection (same_column
    rule). Driving L1 with strong constant input should make Mi1 spike."""
    graph = _build_graph()
    lif = LIFNetwork(graph, LIFParams(dt=0.005))
    rt = RateTracker(graph.n_units, dt=0.005)

    external = np.zeros(graph.n_units)
    external[graph.type_indices["L1"]] = 5.0
    external[graph.type_indices["L2"]] = 5.0
    external[graph.type_indices["L3"]] = 5.0

    for _ in range(300):
        spikes = lif.step(external)
        rates = rt.update(lif.instantaneous_signal())

    assert rates[graph.type_indices["L1"]].mean() > 5.0
    assert rates[graph.type_indices["Mi1"]].mean() > 5.0
    assert np.isfinite(lif.v).all()


def test_hs_vs_are_graded_not_spiking():
    graph = _build_graph()
    lif = LIFNetwork(graph, LIFParams(dt=0.005))
    for t in ("HSN", "HSE", "HSS", "VS1", "VS4", "VS6"):
        assert lif.graded_mask[graph.type_indices[t]].all()


def test_direction_selectivity_emerges_from_real_spatial_offset_and_delay():
    """A rightward-drifting grating should drive T4a (front-to-back
    preferring) more than T4b (back-to-front), and vice versa for a
    leftward grating -- this only works because of the real spatial offset
    (graph.py) *and* the synaptic delay (lif.py); see their docstrings."""
    graph = _build_graph(n_columns=70)

    def mean_rate(speed, cell_type, steps=500, dt=0.005, period=3.0):
        lif = LIFNetwork(graph, LIFParams(dt=dt))
        rt = RateTracker(graph.n_units, dt=dt, tau=0.1)
        for i in range(steps):
            t = i * dt
            external = np.zeros(graph.n_units)
            for (cell, col), idx in graph.column_index.items():
                if cell in ("L1", "L2", "L3"):
                    x, _ = graph.column_xy[col]
                    external[idx] = 4.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (x / period - speed * t)))
            spikes = lif.step(external)
            rates = rt.update(lif.instantaneous_signal())
        return float(rates[graph.type_indices[cell_type]].mean())

    right_a, right_b = mean_rate(2.0, "T4a"), mean_rate(2.0, "T4b")
    left_a, left_b = mean_rate(-2.0, "T4a"), mean_rate(-2.0, "T4b")

    assert right_a > right_b
    assert left_b > left_a
