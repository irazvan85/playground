"""Expands the cell-type-level connectivity table into an instance-level,
signed sparse weight matrix over every modeled neuron.

Three wiring rules, matching how `data/connectome/connectivity_table.csv`
describes each edge:

- ``same_column``: retinotopic connections in the early visual pathway
  (lamina -> medulla -> T4/T5). Each real neuron instance connects only to
  the instance of the target type in the *same* real hex column.
- ``neighbor_offset``: the null-direction-suppression inputs to T4/T5
  (Haag, Arenz, Serbe & Borst 2016). The presynaptic partner is looked up in
  a real neighboring hex column, offset from the postsynaptic column by
  (offset_dx, offset_dy) hex-lattice steps -- found via nearest-neighbor
  search over the *real* x/y column coordinates from the vendored
  Matsliah2024 data. These edges are routed through a separate, slower
  synaptic time constant in `model/lif.py` (a delay line), because spatial
  offset alone is not sufficient for direction selectivity -- the classic
  Hassenstein-Reichardt/Barlow-Levick correlator needs the antagonist input
  to arrive *delayed* relative to the direct one, so it only lines up (and
  cancels) for null-direction motion. This is exactly the mechanism
  Behnia et al. 2014 measured (different temporal filtering on the two
  input channels) and Haag et al. 2016 modeled. Without both the spatial
  offset *and* the delay, T4a and T4b would respond identically.
- ``pool_all``: wide-field integration (T4/T5 -> HS/VS, HS/VS -> descending
  neurons). Every instance of the presynaptic type connects to every
  instance of the (non-columnar, singleton) postsynaptic type, with the
  edge weight normalized by fan-in so total drive stays comparable across
  differently-sized column patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .constants import SINGLETON_TYPES
from .loader import ConnectomeData


@dataclass
class InstanceGraph:
    n_units: int
    weight_fast: sp.csr_matrix  # same_column + pool_all edges; weight[post, pre]
    weight_delayed: sp.csr_matrix  # neighbor_offset (delay-line) edges; weight[post, pre]
    unit_type: np.ndarray
    unit_column: np.ndarray  # -1 for non-columnar (singleton) units
    unit_root_id: np.ndarray  # -1 for non-columnar units (no single real root_id)
    type_indices: Dict[str, np.ndarray]
    column_index: Dict[Tuple[str, int], int]
    column_xy: Dict[int, Tuple[float, float]]
    neighbor_spacing: float

    @property
    def weight(self) -> sp.csr_matrix:
        """Combined (fast + delayed) weight matrix, e.g. for structural
        inspection/tests. The simulator itself uses `weight_fast` and
        `weight_delayed` separately so the delay line has an effect.
        """
        return self.weight_fast + self.weight_delayed


def _median_neighbor_spacing(column_xy: Dict[int, Tuple[float, float]]) -> float:
    if len(column_xy) < 2:
        return 1.0
    pts = np.array(list(column_xy.values()))
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2)
    return float(np.median(dist[:, 1]))


def build_instance_graph(data: ConnectomeData) -> InstanceGraph:
    neurons = data.neurons
    connectivity = data.connectivity

    unit_type: List[str] = []
    unit_column: List[int] = []
    unit_root_id: List[int] = []
    type_indices: Dict[str, List[int]] = {}
    column_index: Dict[Tuple[str, int], int] = {}

    def add_unit(t: str, column_id, root_id) -> int:
        idx = len(unit_type)
        unit_type.append(t)
        unit_column.append(int(column_id) if column_id is not None else -1)
        unit_root_id.append(int(root_id) if root_id is not None else -1)
        type_indices.setdefault(t, []).append(idx)
        if column_id is not None:
            column_index[(t, int(column_id))] = idx
        return idx

    for row in neurons.itertuples(index=False):
        add_unit(row.type, row.column_id, row.root_id)

    for t in SINGLETON_TYPES:
        add_unit(t, None, None)

    column_xy: Dict[int, Tuple[float, float]] = {
        int(c): (float(x), float(y))
        for c, x, y in neurons.groupby("column_id")[["x", "y"]]
        .mean()
        .reset_index()[["column_id", "x", "y"]]
        .itertuples(index=False)
    }
    spacing = _median_neighbor_spacing(column_xy)

    n_units = len(unit_type)
    fast_rows: List[int] = []
    fast_cols: List[int] = []
    fast_vals: List[float] = []
    delayed_rows: List[int] = []
    delayed_cols: List[int] = []
    delayed_vals: List[float] = []

    for edge in connectivity.itertuples(index=False):
        signed_w = edge.weight if edge.sign == "excitatory" else -edge.weight

        if edge.spatial_rule == "same_column":
            pre_by_col = {
                c: i for (t, c), i in column_index.items() if t == edge.pre_type
            }
            post_by_col = {
                c: i for (t, c), i in column_index.items() if t == edge.post_type
            }
            for c, pre_idx in pre_by_col.items():
                post_idx = post_by_col.get(c)
                if post_idx is not None:
                    fast_rows.append(post_idx)
                    fast_cols.append(pre_idx)
                    fast_vals.append(signed_w)

        elif edge.spatial_rule == "neighbor_offset":
            pre_cols = [c for (t, c) in column_index if t == edge.pre_type]
            if not pre_cols:
                continue
            pre_xy = np.array([column_xy[c] for c in pre_cols])
            tree = cKDTree(pre_xy)

            post_items = [
                (c, i) for (t, c), i in column_index.items() if t == edge.post_type
            ]
            for c, post_idx in post_items:
                cx, cy = column_xy[c]
                target = (
                    cx + edge.offset_dx * spacing,
                    cy + edge.offset_dy * spacing,
                )
                _, j = tree.query(target)
                pre_col = pre_cols[j]
                pre_idx = column_index[(edge.pre_type, pre_col)]
                delayed_rows.append(post_idx)
                delayed_cols.append(pre_idx)
                delayed_vals.append(signed_w)

        elif edge.spatial_rule == "pool_all":
            pre_idxs = type_indices.get(edge.pre_type, [])
            post_idxs = type_indices.get(edge.post_type, [])
            if not pre_idxs or not post_idxs:
                continue
            scaled_w = signed_w / max(len(pre_idxs), 1)
            for pre_idx in pre_idxs:
                for post_idx in post_idxs:
                    fast_rows.append(post_idx)
                    fast_cols.append(pre_idx)
                    fast_vals.append(scaled_w)
        else:
            raise ValueError(f"unknown spatial_rule {edge.spatial_rule!r}")

    weight_fast = sp.csr_matrix((fast_vals, (fast_rows, fast_cols)), shape=(n_units, n_units))
    weight_delayed = sp.csr_matrix(
        (delayed_vals, (delayed_rows, delayed_cols)), shape=(n_units, n_units)
    )

    return InstanceGraph(
        n_units=n_units,
        weight_fast=weight_fast,
        weight_delayed=weight_delayed,
        unit_type=np.array(unit_type),
        unit_column=np.array(unit_column),
        unit_root_id=np.array(unit_root_id),
        type_indices={t: np.array(idxs) for t, idxs in type_indices.items()},
        column_index=column_index,
        column_xy=column_xy,
        neighbor_spacing=spacing,
    )
