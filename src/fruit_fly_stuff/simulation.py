"""Discrete-time, sparse, spike-based toy circuit with reward-modulated plasticity.

The update equations are intentionally simple and are NOT a validated model of
fly neuron dynamics. A connectome specifies topology, not all physiological
parameters. This engine is meant for correctness tests and small experiments.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Connectome:
    """Directed edge list. One aggregated connection per (source, target) pair.

    weights may be negative for fixed inhibitory connections. Plastic weights
    must be nonnegative; a more complete model would distinguish receptor and
    neurotransmitter classes and use biologically calibrated plasticity rules.
    """

    n_neurons: int
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    plastic: np.ndarray

    def __post_init__(self) -> None:
        if isinstance(self.n_neurons, bool) or not isinstance(self.n_neurons, (int, np.integer)) or self.n_neurons < 1:
            raise ValueError("n_neurons must be a positive integer")
        arrays = {
            "source": np.asarray(self.source),
            "target": np.asarray(self.target),
            "weight": np.asarray(self.weight, dtype=np.float64),
            "plastic": np.asarray(self.plastic),
        }
        if any(arr.ndim != 1 for arr in arrays.values()):
            raise ValueError("edge arrays must be one-dimensional")
        if len({len(arr) for arr in arrays.values()}) != 1:
            raise ValueError("edge arrays must have equal lengths")
        for name in ("source", "target"):
            arr = arrays[name]
            if arr.dtype.kind not in "iu" or np.any(arr < 0) or np.any(arr >= self.n_neurons):
                raise ValueError(f"{name} must contain valid integer neuron indices")
            arrays[name] = arr.astype(np.int64, copy=True)
        if not np.all(np.isfinite(arrays["weight"])):
            raise ValueError("weights must be finite")
        if arrays["plastic"].dtype.kind != "b":
            raise ValueError("plastic must be a boolean array")
        if np.any(arrays["plastic"] & (arrays["weight"] < 0)):
            raise ValueError("plastic weights must be nonnegative")
        # Standardize ordering by target (CSR row), then source (CSR column).
        order = np.lexsort((arrays["source"], arrays["target"]))
        for name, arr in arrays.items():
            object.__setattr__(self, name, arr[order].copy())
        if len(order) > 1 and np.any(
            (self.source[1:] == self.source[:-1])
            & (self.target[1:] == self.target[:-1])
        ):
            raise ValueError("duplicate neuron pairs must be aggregated before import")


@dataclass(frozen=True)
class SimulationConfig:
    """All times are discrete simulation steps, not calibrated milliseconds."""

    leak: float = 0.5
    threshold: float = 1.0
    eligibility_decay: float = 0.85
    learning_rate: float = 0.18
    max_plastic_weight: float = 2.0

    def __post_init__(self) -> None:
        values = (self.leak, self.threshold, self.eligibility_decay,
                  self.learning_rate, self.max_plastic_weight)
        if not all(np.isfinite(x) for x in values):
            raise ValueError("simulation parameters must be finite")
        if not (0 <= self.leak < 1 and 0 <= self.eligibility_decay <= 1):
            raise ValueError("leak must be in [0,1); eligibility_decay in [0,1]")
        if self.threshold <= 0 or self.learning_rate < 0 or self.max_plastic_weight <= 0:
            raise ValueError("threshold and max_plastic_weight must be positive; learning rate nonnegative")


class SpikingBrain:
    """Synchronous LIF-inspired update with delayed spikes and eligibility traces.

    A spike generated at t contributes to downstream inputs at t+1.
    The eligibility trace of a plastic edge increases when the source fired
    at t-1 and the target fires at t. Reward updates eligible synapses.
    """

    def __init__(self, connectome: Connectome, config: SimulationConfig | None = None):
        self.connectome = connectome
        self.config = config or SimulationConfig()
        counts = np.bincount(connectome.target, minlength=connectome.n_neurons)
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.weights = connectome.weight.copy()
        # Edges are sorted in Connectome, so CSR data positions match self.weights.
        self._matrix = csr_matrix(
            (self.weights, connectome.source.copy(), indptr),
            shape=(connectome.n_neurons, connectome.n_neurons),
            copy=False,
        )
        self._plastic_indices = np.flatnonzero(connectome.plastic)
        if np.any(self.weights[self._plastic_indices] > self.config.max_plastic_weight):
            raise ValueError("initial plastic weights exceed max_plastic_weight")
        self.potential = np.zeros(connectome.n_neurons, dtype=np.float64)
        self.previous_spikes = np.zeros(connectome.n_neurons, dtype=bool)
        self.eligibility = np.zeros(len(self._plastic_indices), dtype=np.float64)
        self.steps = 0

    def reset_activity(self, *, clear_eligibility: bool = True) -> None:
        """Reset short-term state between trials, preserving learned weights."""
        self.potential.fill(0)
        self.previous_spikes.fill(False)
        if clear_eligibility:
            self.eligibility.fill(0)

    def step(self, stimulus: np.ndarray | None = None, *, reward: float = 0.0) -> np.ndarray:
        """Advance one step and return a copy of the newly fired spike vector."""
        if stimulus is None:
            drive = np.zeros(self.connectome.n_neurons, dtype=np.float64)
        else:
            drive = np.asarray(stimulus, dtype=np.float64)
            if drive.shape != (self.connectome.n_neurons,) or not np.all(np.isfinite(drive)):
                raise ValueError("stimulus must be a finite vector of length n_neurons")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

        # Keep matrix.data and weights referring to the same data buffer.
        synaptic_current = self._matrix @ self.previous_spikes.astype(np.float64)
        self.potential = self.config.leak * self.potential + synaptic_current + drive
        fired = self.potential >= self.config.threshold
        self.potential[fired] = 0.0

        if len(self._plastic_indices):
            idx = self._plastic_indices
            coincidence = (self.previous_spikes[self.connectome.source[idx]]
                           & fired[self.connectome.target[idx]])
            self.eligibility *= self.config.eligibility_decay
            self.eligibility += coincidence
            if reward != 0.0:
                self.weights[idx] = np.clip(
                    self.weights[idx] + self.config.learning_rate * reward * self.eligibility,
                    0.0,
                    self.config.max_plastic_weight,
                )
        self.previous_spikes = fired.copy()
        self.steps += 1
        return fired.copy()
