#!/usr/bin/env python3
"""How to replace this project's literature-derived weights with a true,
live per-synapse pull from neuprint -- **not runnable inside this repo's
development sandbox**, where neuprint.janelia.org is blocked by network
policy (confirmed HTTP 403; see data/connectome/SOURCES.md).

Run this outside that sandbox, with `pip install neuprint-python` and a
real neuprint API token (get one at https://neuprint.janelia.org after
logging in, from your account page):

    python scripts/fetch_connectome_data.py --token <your-neuprint-token>

This queries real per-synapse-count connectivity for the same cell types
`data/connectome/connectivity_table.csv` currently encodes with literature
citations, and writes it out in the same schema so
`connectome/loader.py`'s `BundledLiteratureSource` (or a new
`LiveNeuprintSource` you finish implementing per its docstring in
`connectome/loader.py`) can consume it without any other code changes.
"""

from __future__ import annotations

import argparse
import sys

CELL_TYPES = [
    "L1", "L2", "L3", "Mi1", "Mi4", "Mi9", "Tm1", "Tm4", "Tm9",
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", required=True, help="neuprint API token")
    parser.add_argument("--dataset", default="hemibrain:v1.2.1")
    parser.add_argument("--out", default="data/connectome/live_connectivity.csv")
    args = parser.parse_args()

    try:
        from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies
    except ImportError:
        print(
            "neuprint-python is not installed. Run: pip install neuprint-python",
            file=sys.stderr,
        )
        raise SystemExit(1)

    client = Client("neuprint.janelia.org", dataset=args.dataset, token=args.token)
    criteria = NC(type=CELL_TYPES, client=client)
    _neuron_df, conn_df = fetch_adjacencies(criteria, criteria, client=client)

    # conn_df has real per-partner synapse-count `weight` columns keyed by
    # individual bodyId pairs. Aggregate to the cell-type level to match
    # connectivity_table.csv's schema (or skip aggregation and build the
    # instance graph directly from real per-neuron pairs -- see the
    # LiveNeuprintSource docstring in connectome/loader.py for that path).
    type_lookup = _neuron_df.set_index("bodyId")["type"]
    conn_df["pre_type"] = conn_df["bodyId_pre"].map(type_lookup)
    conn_df["post_type"] = conn_df["bodyId_post"].map(type_lookup)
    aggregated = (
        conn_df.groupby(["pre_type", "post_type"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "synapse_count"})
    )
    aggregated.to_csv(args.out, index=False)
    print(f"Wrote real per-cell-type synapse counts for {len(aggregated)} edges to {args.out}")


if __name__ == "__main__":
    main()
