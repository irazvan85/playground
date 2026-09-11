"""Maps the readout's raw output to a safety-clipped drone command."""

from __future__ import annotations

import numpy as np


def to_command(raw: np.ndarray, hover_thrust: float, forward_thrust: float = 0.5) -> np.ndarray:
    """raw = [roll, pitch, yaw, thrust_delta], each expected roughly in
    [-1, 1] from the readout's tanh output. Vertical thrust is expressed as
    a delta around a hover baseline so the readout only has to correct
    around equilibrium, not learn to reproduce it from scratch. Forward
    thrust is held at a fixed baseline (see DroneDynamics docstring) -- the
    readout's job is steering (roll/pitch/yaw), not propulsion.
    """
    roll, pitch, yaw, thrust_delta = raw
    thrust = np.clip(hover_thrust + 0.4 * thrust_delta, 0.0, 1.0)
    return np.array(
        [
            np.clip(roll, -1.0, 1.0),
            np.clip(pitch, -1.0, 1.0),
            np.clip(yaw, -1.0, 1.0),
            thrust,
            np.clip(forward_thrust, 0.0, 1.0),
        ]
    )
