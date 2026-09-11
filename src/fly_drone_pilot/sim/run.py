"""Closed loop: environment -> compound eye -> connectome-derived LIF network
-> descending-neuron readout -> drone controller -> drone dynamics -> (new
position/orientation feeds back into the eye next step).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..connectome.constants import DN_TYPES
from ..connectome.graph import InstanceGraph, build_instance_graph
from ..connectome.loader import BundledLiteratureSource
from ..drone.controller import to_command
from ..drone.dynamics import DroneDynamics, DroneParams, DroneState
from ..model.lif import LIFNetwork, LIFParams, RateTracker
from ..model.readout import Readout
from ..vision.optic_flow import CompoundEye, photoreceptor_drive
from .environment import Environment


def build_graph(n_columns: int = 150, center=None) -> InstanceGraph:
    source = BundledLiteratureSource(n_columns=n_columns, center=center)
    return build_instance_graph(source.load())


def rates_by_type(rate_vector: np.ndarray, graph: InstanceGraph, types) -> dict:
    out = {}
    for t in types:
        idxs = graph.type_indices.get(t)
        out[t] = float(np.mean(rate_vector[idxs])) if idxs is not None and len(idxs) else 0.0
    return out


@dataclass
class EpisodeResult:
    forward_progress: float
    collided: bool
    min_clearance: float
    n_steps_survived: int
    mean_altitude_error: float = 0.0
    trajectory: Optional[np.ndarray] = None
    spike_raster: Optional[np.ndarray] = None
    dn_rate_history: Optional[np.ndarray] = None
    luminance_snapshot: Optional[dict] = None


def run_episode(
    graph: InstanceGraph,
    readout: Readout,
    n_steps: int = 300,
    dt: float = 0.005,
    seed: int = 0,
    record: bool = False,
) -> EpisodeResult:
    lif = LIFNetwork(graph, LIFParams(dt=dt))
    rate_tracker = RateTracker(graph.n_units, dt=dt)
    environment = Environment(seed=seed)
    eye = CompoundEye(graph.column_xy)
    drone_params = DroneParams()
    hover_thrust = drone_params.mass * drone_params.gravity / drone_params.max_thrust
    drone = DroneDynamics(drone_params)
    start_altitude = environment.ceiling / 2.0
    drone.reset(
        DroneState(position=np.array([1.0, 0.0, start_altitude]))
    )

    trajectory = np.zeros((n_steps, 3)) if record else None
    spike_raster = np.zeros((n_steps, graph.n_units), dtype=bool) if record else None
    dn_history = np.zeros((n_steps, len(DN_TYPES))) if record else None
    luminance_snapshot = None

    min_clearance = np.inf
    collided = False
    n_survived = n_steps
    altitude_error_sum = 0.0

    for step in range(n_steps):
        luminance = eye.sample(environment, drone.state.position, drone.state.quat)
        if record and step == n_steps // 2:
            luminance_snapshot = dict(luminance)

        external = photoreceptor_drive(graph, luminance)
        spikes = lif.step(external)
        rates = rate_tracker.update(lif.instantaneous_signal())

        dn_rates = rates_by_type(rates, graph, DN_TYPES)
        raw_command = readout(dn_rates)
        command = to_command(raw_command, hover_thrust=hover_thrust)

        drone.step(command, dt)

        clearance = environment.clearance(drone.state.position)
        min_clearance = min(min_clearance, clearance)
        altitude_error_sum += abs(drone.state.position[2] - start_altitude)

        if record:
            trajectory[step] = drone.state.position
            spike_raster[step] = spikes
            dn_history[step] = [dn_rates[t] for t in DN_TYPES]

        if clearance <= 0:
            collided = True
            n_survived = step + 1
            if record:
                trajectory[step + 1 :] = drone.state.position
            break

    forward_progress = float(drone.state.position[0] - 1.0)
    mean_altitude_error = altitude_error_sum / max(n_survived, 1)

    return EpisodeResult(
        forward_progress=forward_progress,
        collided=collided,
        min_clearance=float(min_clearance),
        n_steps_survived=n_survived,
        mean_altitude_error=mean_altitude_error,
        trajectory=trajectory,
        spike_raster=spike_raster,
        dn_rate_history=dn_history,
        luminance_snapshot=luminance_snapshot,
    )
