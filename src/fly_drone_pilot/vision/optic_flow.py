"""The fly's compound eye, built from the *real* retinotopic hex-column
positions in the vendored FlyWire optic-lobe data (Matsliah et al. 2024).

This module only samples raw scene luminance per real ommatidial column as
the drone moves and rotates -- it computes no motion or flow itself. Motion
selectivity is a property of the connectome-derived T4/T5 wiring
(`connectome/graph.py`'s `same_column` + `neighbor_offset` rules), exactly as
in the real fly: photoreceptors report brightness, and direction-selective
cells downstream compare neighboring columns over time.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..connectome.graph import InstanceGraph
from ..drone.dynamics import quat_to_rotmat


class CompoundEye:
    """Maps each real hex column's (x, y) retinotopic position to a body-frame
    viewing direction, spanning a forward-facing field of view.
    """

    def __init__(
        self,
        column_xy: Dict[int, Tuple[float, float]],
        fov_azimuth_deg: float = 100.0,
        fov_elevation_deg: float = 70.0,
    ):
        cols = np.array(list(column_xy.keys()))
        xy = np.array(list(column_xy.values()))
        x, y = xy[:, 0], xy[:, 1]
        x_span = (x.max() - x.min()) or 1.0
        y_span = (y.max() - y.min()) or 1.0
        xn = (x - x.min()) / x_span - 0.5
        yn = (y - y.min()) / y_span - 0.5
        az = np.radians(fov_azimuth_deg) * xn
        el = np.radians(fov_elevation_deg) * yn
        directions = np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1
        )
        self.directions: Dict[int, np.ndarray] = {
            int(c): directions[i] for i, c in enumerate(cols)
        }

    def sample(self, environment, position: np.ndarray, quat: np.ndarray, max_range: float = 20.0) -> Dict[int, float]:
        rot = quat_to_rotmat(quat)
        out: Dict[int, float] = {}
        for column_id, direction_body in self.directions.items():
            direction_world = rot @ direction_body
            _, luminance = environment.raycast(position, direction_world, max_range=max_range)
            out[column_id] = luminance
        return out


def photoreceptor_drive(
    graph: InstanceGraph,
    luminance_by_column: Dict[int, float],
    gain: float = 4.0,
    driven_types: Tuple[str, ...] = ("L1", "L2", "L3"),
) -> np.ndarray:
    """Builds the LIF network's external input vector: raw luminance (scaled
    by `gain`) injected into the real lamina monopolar cell instances
    (L1/L2/L3), zero everywhere else.
    """
    external = np.zeros(graph.n_units)
    for (cell_type, column_id), unit_idx in graph.column_index.items():
        if cell_type in driven_types:
            luminance = luminance_by_column.get(column_id)
            if luminance is not None:
                external[unit_idx] = gain * luminance
    return external
