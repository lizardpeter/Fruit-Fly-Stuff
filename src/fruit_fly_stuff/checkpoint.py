"""Atomic, non-pickle checkpoints for reproducible simulation restarts."""

from pathlib import Path
import os
import tempfile

import numpy as np

from .simulation import Connectome, SimulationConfig, SpikingBrain

SCHEMA_VERSION = 1
_CONFIG_FIELDS = ("leak", "threshold", "eligibility_decay", "learning_rate", "max_plastic_weight")
_REQUIRED = {"version", "n_neurons", "source", "target", "weights", "plastic",
             "potential", "previous_spikes", "eligibility", "steps", "config"}


def save_checkpoint(brain: SpikingBrain, path: str | Path) -> None:
    """Write full trainable/stateful model, atomically replacing an old snapshot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": np.array([SCHEMA_VERSION], dtype=np.int64),
        "n_neurons": np.array([brain.connectome.n_neurons], dtype=np.int64),
        "source": brain.connectome.source,
        "target": brain.connectome.target,
        "weights": brain.weights,
        "plastic": brain.connectome.plastic,
        "potential": brain.potential,
        "previous_spikes": brain.previous_spikes,
        "eligibility": brain.eligibility,
        "steps": np.array([brain.steps], dtype=np.int64),
        "config": np.array([getattr(brain.config, field) for field in _CONFIG_FIELDS]),
    }
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".checkpoint-", suffix=".npz", delete=False) as temp:
            temp_name = temp.name
            np.savez_compressed(temp, **data)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, output)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def load_checkpoint(path: str | Path) -> SpikingBrain:
    """Restore weights AND temporal state; never execute pickle payloads."""
    with np.load(path, allow_pickle=False) as archive:
        if not _REQUIRED.issubset(archive.files):
            raise ValueError("checkpoint is missing required arrays")
        if archive["version"].shape != (1,) or int(archive["version"][0]) != SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint version")
        if archive["n_neurons"].shape != (1,) or archive["steps"].shape != (1,):
            raise ValueError("invalid checkpoint metadata")
        config_values = archive["config"]
        if config_values.shape != (len(_CONFIG_FIELDS),):
            raise ValueError("invalid checkpoint configuration")
        cfg = SimulationConfig(**dict(zip(_CONFIG_FIELDS, config_values.tolist())))
        network = Connectome(
            n_neurons=int(archive["n_neurons"][0]),
            source=archive["source"],
            target=archive["target"],
            weight=archive["weights"],
            plastic=archive["plastic"],
        )
        brain = SpikingBrain(network, cfg)
        potential = np.asarray(archive["potential"], dtype=np.float64)
        spikes = np.asarray(archive["previous_spikes"])
        eligibility = np.asarray(archive["eligibility"], dtype=np.float64)
        steps = int(archive["steps"][0])
        if (potential.shape != (network.n_neurons,) or
            spikes.shape != (network.n_neurons,) or spikes.dtype.kind != "b" or
            eligibility.shape != (len(brain.eligibility),) or
            not np.all(np.isfinite(potential)) or
            not np.all(np.isfinite(eligibility)) or steps < 0):
            raise ValueError("invalid checkpoint state")
        brain.potential[:] = potential
        brain.previous_spikes[:] = spikes
        brain.eligibility[:] = eligibility
        brain.steps = steps
        return brain
