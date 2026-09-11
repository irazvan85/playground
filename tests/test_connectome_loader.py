import numpy as np

from fly_drone_pilot.connectome.constants import DN_TYPES, LPTC_TYPES
from fly_drone_pilot.connectome.graph import build_instance_graph
from fly_drone_pilot.connectome.loader import BundledLiteratureSource


def test_loads_real_neuron_instances():
    data = BundledLiteratureSource(n_columns=40).load()
    assert len(data.neurons) > 0
    # every row must be a real, individually identified FlyWire neuron
    assert (data.neurons["root_id"] > 0).all()
    assert data.neurons["column_id"].nunique() <= 40
    assert set(data.neurons["type"]) <= {
        "L1", "L2", "L3", "Mi1", "Mi4", "Mi9", "Tm1", "Tm4", "Tm9",
        "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    }


def test_connectivity_table_has_citations():
    data = BundledLiteratureSource(n_columns=10).load()
    conn = data.connectivity
    assert len(conn) > 0
    assert conn["citation_key"].notna().all()
    assert set(conn["sign"]) <= {"excitatory", "inhibitory"}
    assert set(conn["spatial_rule"]) <= {"same_column", "neighbor_offset", "pool_all"}


def test_instance_graph_includes_singleton_units():
    data = BundledLiteratureSource(n_columns=40).load()
    graph = build_instance_graph(data)
    for t in LPTC_TYPES + DN_TYPES:
        assert t in graph.type_indices
        assert len(graph.type_indices[t]) == 1
    assert graph.n_units == len(data.neurons) + len(LPTC_TYPES) + len(DN_TYPES)


def test_neighbor_offset_uses_a_different_column_than_same_column():
    """The null-direction-suppression wiring must not degenerate into
    same-column wiring, or T4a/T4b would be indistinguishable (see
    connectome/graph.py's module docstring)."""
    data = BundledLiteratureSource(n_columns=60).load()
    graph = build_instance_graph(data)

    mi9_to_t4a = [
        (edge.pre_type, edge.post_type)
        for edge in data.connectivity.itertuples(index=False)
        if edge.pre_type == "Mi9" and edge.post_type == "T4a"
    ]
    assert mi9_to_t4a, "expected a Mi9->T4a null-direction-suppression edge"

    # at least some T4a units should receive their Mi9 input from a
    # different column than their own (i.e. real spatial offset happened)
    same_column_count = 0
    total = 0
    weight_delayed = graph.weight_delayed.tocoo()
    t4a_idx = set(graph.type_indices["T4a"].tolist())
    for post, pre, _ in zip(weight_delayed.row, weight_delayed.col, weight_delayed.data):
        if post in t4a_idx:
            total += 1
            if graph.unit_column[post] == graph.unit_column[pre]:
                same_column_count += 1
    assert total > 0
    assert same_column_count < total
