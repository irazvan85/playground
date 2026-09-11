"""Loads the connectome this project's spiking network is built from.

Two sources implement the same `ConnectomeSource` interface:

- `BundledLiteratureSource` (default): real neuron identities and hex-column
  anatomy from Nern et al. 2024 / Matsliah et al. 2024, vendored under
  data/connectome/raw/, paired with literature-cited connection strengths in
  data/connectome/connectivity_table.csv. See data/connectome/SOURCES.md for
  full citations and why this is the default -- neuprint.janelia.org and
  codex.flywire.ai (the live connectome databases) both return HTTP 403 from
  this sandbox's network policy, so a live per-synapse pull isn't possible
  here.
- `LiveNeuprintSource` (stub): documents how to swap in a true per-synapse
  live pull outside this sandbox.

Everything downstream (`graph.py` and beyond) only depends on the
`ConnectomeData` shape this interface returns, so switching sources later
requires no other code changes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from .constants import SINGLETON_TYPES

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "connectome"


@dataclass
class ConnectomeData:
    """neurons: one row per real modeled neuron instance (root_id, type, column_id, x, y).
    connectivity: cell-type-level edges (pre_type, post_type, weight, sign, spatial_rule, citation_key).
    """

    neurons: pd.DataFrame
    connectivity: pd.DataFrame


class ConnectomeSource(abc.ABC):
    @abc.abstractmethod
    def load(self) -> ConnectomeData:
        raise NotImplementedError


class BundledLiteratureSource(ConnectomeSource):
    """Real Nern2024/Matsliah2024 neuron identities/anatomy + literature-cited weights.

    Parameters
    ----------
    data_dir:
        Directory containing `connectivity_table.csv` and `raw/`.
    n_columns:
        How many real hex columns (out of the ~796 present in the vendored
        FlyWire optic-lobe data) to instantiate. Using all of them is
        physically real but produces a network of ~15-20k units; a smaller
        contiguous patch nearest the visual-field center keeps simulation
        fast for the demo while still using genuine per-column neuron
        identities and positions for every unit that is modeled.
    center:
        (x, y) hex-coordinate center to select the column patch around.
        Defaults to the centroid of all used columns.
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        n_columns: int = 150,
        center: Optional[Tuple[float, float]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.n_columns = n_columns
        self.center = center

    def load(self) -> ConnectomeData:
        connectivity = pd.read_csv(self.data_dir / "connectivity_table.csv")
        columnar_types = (
            set(connectivity["pre_type"]) | set(connectivity["post_type"])
        ) - set(SINGLETON_TYPES)

        neurons = pd.read_csv(
            self.data_dir / "raw" / "matsliah2024_fafb_vis_cols.csv"
        )
        neurons = neurons[neurons["type"].isin(columnar_types)].copy()

        col_pos = neurons.groupby("column_id")[["x", "y"]].mean()
        if self.center is None:
            cx, cy = col_pos["x"].mean(), col_pos["y"].mean()
        else:
            cx, cy = self.center
        dist = ((col_pos["x"] - cx) ** 2 + (col_pos["y"] - cy) ** 2) ** 0.5
        chosen_columns = set(dist.sort_values().index[: self.n_columns])

        neurons = neurons[neurons["column_id"].isin(chosen_columns)].copy()
        neurons = neurons[["root_id", "type", "column_id", "x", "y"]].reset_index(
            drop=True
        )

        return ConnectomeData(neurons=neurons, connectivity=connectivity)


class LiveNeuprintSource(ConnectomeSource):
    """Documents how to replace the literature weight table with a true,
    live per-synapse-count pull. **Not usable inside this sandbox**:
    neuprint.janelia.org is blocked by the egress policy (confirmed 403).

    Outside this sandbox, with a real neuprint API token, this would become
    e.g.:

        from neuprint import Client, fetch_adjacencies, NeuronCriteria as NC
        client = Client('neuprint.janelia.org', dataset=self.dataset, token=self.token)
        types = ['L1','L2','L3','L4','L5','Mi1','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9',
                 'T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']
        neuron_df, conn_df = fetch_adjacencies(NC(type=types), NC(type=types), client=client)
        # conn_df has real per-partner `weight` (synapse counts) which would
        # replace the hand-curated `weight`/`sign` columns in
        # connectivity_table.csv, keyed by real (pre, post) neuron pairs
        # instead of by cell type.

    `graph.py` would then build the instance graph directly from `conn_df`
    rather than expanding a cell-type table via `same_column`/`pool_all`
    rules -- no other module needs to change.
    """

    def __init__(self, token: str, dataset: str = "hemibrain:v1.2.1"):
        self.token = token
        self.dataset = dataset

    def load(self) -> ConnectomeData:
        raise NotImplementedError(
            "Live neuprint access is blocked by this sandbox's network policy "
            "(neuprint.janelia.org -> HTTP 403, 'policy denial'). Run this "
            "outside the sandbox with a real neuprint API token; see this "
            "class's docstring for the exact calls."
        )
