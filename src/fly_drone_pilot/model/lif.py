"""Vectorized leaky-integrate-and-fire (LIF) spiking simulation over the
connectome-derived instance graph.

Every modeled neuron (real, identified visual-pathway cells plus the
wide-field/descending/motor singleton units) is a leaky integrator driven
through the signed, connectome-derived sparse weight matrices built by
`connectome.graph.build_instance_graph`; only the biological weights the
connectome/literature data justify are used -- there is no generic
dense/random connectivity anywhere in this network.

Two synaptic channels are simulated per Haag et al. 2016 / Behnia et al.
2014's account of how direction selectivity actually arises: a fast channel
(same-column and wide-field-pooling edges) and a slower, delayed channel
(the spatially-offset null-direction-suppression edges). The delay -- not
just the spatial offset -- is what makes T4/T5 subtypes direction-selective;
see `connectome/graph.py` for the wiring-rule side of this.

Most units (the early visual pathway, T4/T5) spike with a hard threshold.
The lobula plate tangential cells (HS/VS) are modeled as **graded, non-
spiking** units instead -- this is not a simplification of convenience, it
is how these specific real cells actually work (Haag & Borst and others
have repeatedly shown HS/VS respond with graded membrane potential shifts,
not action potentials). The descending-neuron pool is also treated as
graded here, as a modeling simplification (see data/connectome/SOURCES.md)
so a clean continuous drive signal reaches the readout without depending on
a spike threshold three synapses downstream of the graded LPTCs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..connectome.constants import DN_TYPES, LPTC_TYPES
from ..connectome.graph import InstanceGraph


@dataclass
class LIFParams:
    dt: float = 1e-3  # s
    tau_m: float = 0.02  # membrane time constant, s
    tau_syn_fast: float = 0.005  # fast (direct) synaptic channel time constant, s
    tau_syn_delayed: float = 0.04  # delayed (null-direction antagonist) channel time constant, s
    v_th: float = 0.35
    v_reset: float = 0.0
    refractory_steps: int = 2
    input_gain: float = 1.0
    synaptic_gain: float = 2.5  # scales connectome weights -> postsynaptic potential size


class LIFNetwork:
    """Discrete-time leaky-integrator network of shape (n_units,)."""

    def __init__(self, graph: InstanceGraph, params: LIFParams = None):
        self.graph = graph
        self.params = params or LIFParams()
        n = graph.n_units

        self.graded_mask = np.zeros(n, dtype=bool)
        self.rectify_mask = np.zeros(n, dtype=bool)
        for t in LPTC_TYPES:
            self.graded_mask[graph.type_indices.get(t, [])] = True
        for t in DN_TYPES:
            idx = graph.type_indices.get(t, [])
            self.graded_mask[idx] = True
            self.rectify_mask[idx] = True

        self.v = np.zeros(n)
        self.syn_fast = np.zeros(n)
        self.syn_delayed = np.zeros(n)
        self.refrac = np.zeros(n, dtype=int)
        self.spikes = np.zeros(n, dtype=bool)
        self.activity = np.zeros(n)  # what actually propagates: spike (0/1) or graded potential

        self._fast_decay = float(np.exp(-self.params.dt / self.params.tau_syn_fast))
        self._delayed_decay = float(np.exp(-self.params.dt / self.params.tau_syn_delayed))
        self._mem_alpha = self.params.dt / self.params.tau_m

    def reset(self) -> None:
        self.v[:] = 0.0
        self.syn_fast[:] = 0.0
        self.syn_delayed[:] = 0.0
        self.refrac[:] = 0
        self.spikes[:] = False
        self.activity[:] = 0.0

    def step(self, external_input: np.ndarray) -> np.ndarray:
        """Advance the network by one `dt`, using last step's `activity` as
        the presynaptic drive (causal: yesterday's output affects today's
        input). `external_input` (n_units,) is added directly for driven
        (sensory input) units and should be zero elsewhere. Returns the
        boolean spike vector for this step (all-False for graded units).
        """
        p = self.params

        fast_current = p.synaptic_gain * (self.graph.weight_fast @ self.activity)
        delayed_current = p.synaptic_gain * (self.graph.weight_delayed @ self.activity)
        self.syn_fast = self.syn_fast * self._fast_decay + fast_current
        self.syn_delayed = self.syn_delayed * self._delayed_decay + delayed_current

        total_input = self.syn_fast + self.syn_delayed + p.input_gain * external_input

        active = self.refrac <= 0
        self.v[active] += self._mem_alpha * (-self.v[active] + total_input[active])
        self.v[~active] = p.v_reset
        self.refrac[~active] -= 1

        fired = active & ~self.graded_mask & (self.v >= p.v_th)
        self.v[fired] = p.v_reset
        self.refrac[fired] = p.refractory_steps
        self.spikes = fired

        graded_value = np.where(self.rectify_mask, np.maximum(self.v, 0.0), self.v)
        self.activity = np.where(self.graded_mask, graded_value, self.spikes.astype(np.float64))

        return fired

    def instantaneous_signal(self) -> np.ndarray:
        """Per-unit instantaneous drive for rate/activity tracking: spikes
        expressed as a rate (1/dt when firing, else 0) for spiking units,
        and the graded potential directly for HS/VS/DN units.
        """
        spike_rate = self.spikes.astype(np.float64) / self.params.dt
        return np.where(self.graded_mask, self.activity, spike_rate)


class RateTracker:
    """Exponential-moving-average estimate of each unit's instantaneous
    signal (firing rate in Hz for spiking units, graded potential for
    graded units -- see `LIFNetwork.instantaneous_signal`).
    """

    def __init__(self, n_units: int, dt: float, tau: float = 0.05):
        self.dt = dt
        self.decay = float(np.exp(-dt / tau))
        self.rate = np.zeros(n_units)

    def update(self, instantaneous: np.ndarray) -> np.ndarray:
        self.rate = self.decay * self.rate + (1 - self.decay) * instantaneous
        return self.rate
