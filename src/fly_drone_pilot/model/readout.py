"""Small readout layer: descending-neuron-pool firing rates -> drone command.

This is the one part of the network that is *not* connectome-derived -- the
connectome only tells us DN_yaw_L/DN_yaw_R/etc. exist and what drives them
(see data/connectome/SOURCES.md); it doesn't tell us the exact linear
combination a downstream flight motor circuit uses to turn that population
code into actuator commands. `train_readout.py` fits these ~32 parameters
against a simple flight-stabilization objective; the connectome weights
upstream of this layer are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np

from ..connectome.constants import READOUT_INPUT_TYPES

N_OUTPUTS = 4  # roll, pitch, yaw, thrust_delta


@dataclass
class Readout:
    weights: np.ndarray = field(default_factory=lambda: np.zeros((N_OUTPUTS, len(READOUT_INPUT_TYPES))))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(N_OUTPUTS))
    input_types: Sequence[str] = field(default_factory=lambda: tuple(READOUT_INPUT_TYPES))

    def __call__(self, dn_rates_by_type: Dict[str, float]) -> np.ndarray:
        x = np.array([dn_rates_by_type.get(t, 0.0) for t in self.input_types])
        raw = self.weights @ x + self.bias
        return np.tanh(raw)

    def flatten(self) -> np.ndarray:
        return np.concatenate([self.weights.ravel(), self.bias.ravel()])

    @classmethod
    def from_flat(cls, flat: np.ndarray, input_types: Sequence[str] = READOUT_INPUT_TYPES) -> "Readout":
        n_in = len(input_types)
        weights = flat[: N_OUTPUTS * n_in].reshape(N_OUTPUTS, n_in)
        bias = flat[N_OUTPUTS * n_in :]
        return cls(weights=weights, bias=bias, input_types=tuple(input_types))

    def save(self, path: str) -> None:
        np.savez(path, weights=self.weights, bias=self.bias, input_types=np.array(self.input_types))

    @classmethod
    def load(cls, path: str) -> "Readout":
        d = np.load(path, allow_pickle=True)
        return cls(weights=d["weights"], bias=d["bias"], input_types=tuple(d["input_types"]))
