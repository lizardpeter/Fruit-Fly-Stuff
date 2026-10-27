"""Reproducible three-neuron reward-conditioning demonstration.

Neurons: 0=cued stimulus, 1=unconditioned reward cue, 2=output.
This is a synthetic circuit, NOT the biological mushroom body.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .simulation import Connectome, SpikingBrain


def make_toy_brain() -> SpikingBrain:
    return SpikingBrain(Connectome(
        n_neurons=3,
        source=np.array([0, 1]),
        target=np.array([2, 2]),
        weight=np.array([0.10, 1.10]),
        plastic=np.array([True, False]),
    ))


def cue_response(brain: SpikingBrain) -> bool:
    """Return whether a cue alone causes the output neuron to spike."""
    brain.reset_activity()
    brain.step(np.array([2.0, 0.0, 0.0]))
    response = brain.step()
    return bool(response[2])


def train(brain: SpikingBrain, trials: int = 6) -> None:
    for _ in range(trials):
        brain.reset_activity()
        brain.step(np.array([2.0, 2.0, 0.0]))  # paired cue and unconditioned input
        brain.step(reward=1.0)                   # output spikes, eligible edge strengthens


def experiment(path: Path, trials: int = 6) -> dict[str, object]:
    if trials < 1:
        raise ValueError("trials must be >=1")
    brain = make_toy_brain()
    before = cue_response(brain)
    train(brain, trials=trials)
    learned_weight = float(brain.weights[0])  # edge ordering: both target 2, source 0 first
    save_checkpoint(brain, path)
    restored = load_checkpoint(path)
    after = cue_response(restored)
    return {
        "model": "synthetic three-neuron conditioning circuit (not a fruit-fly connectome)",
        "trials": trials,
        "before_training_cue_evokes_output": before,
        "after_restoring_cue_evokes_output": after,
        "learned_cue_weight": round(learned_weight, 6),
        "checkpoint": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("state/toy-brain.npz"))
    parser.add_argument("--trials", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(experiment(args.checkpoint, args.trials), indent=2))


if __name__ == "__main__":
    main()
