import numpy as np
import pytest

from fruit_fly_stuff import Connectome, SimulationConfig, SpikingBrain
from fruit_fly_stuff.demo import cue_response, make_toy_brain, train


def test_spikes_arrive_one_step_later():
    brain = make_toy_brain()
    first = brain.step(np.array([2.0, 2.0, 0.0]))
    assert first.tolist() == [True, True, False]
    second = brain.step()
    assert second.tolist() == [False, False, True]


def test_training_changes_behavior_without_external_memory():
    brain = make_toy_brain()
    assert not cue_response(brain)
    train(brain)
    assert brain.weights[0] > 1.0
    assert cue_response(brain)


def test_no_reward_does_not_change_weight():
    brain = make_toy_brain()
    for _ in range(6):
        brain.reset_activity()
        brain.step(np.array([2.0, 2.0, 0.0]))
        brain.step(reward=0.0)
    assert brain.weights[0] == pytest.approx(0.1)
    assert not cue_response(brain)


def test_negative_reward_reduces_plastic_weight():
    brain = make_toy_brain()
    train(brain)
    initial = brain.weights[0]
    brain.reset_activity()
    brain.step(np.array([2.0, 2.0, 0.0]))
    brain.step(reward=-1.0)
    assert brain.weights[0] < initial


def test_connectome_rejects_duplicate_edges():
    with pytest.raises(ValueError, match="duplicate"):
        Connectome(2, np.array([0, 0]), np.array([1, 1]),
                   np.array([0.4, 0.5]), np.array([False, False]))


def test_inhibitory_fixed_edge():
    brain = SpikingBrain(Connectome(2, np.array([0]), np.array([1]),
                                         np.array([-2.0]), np.array([False])),
                         SimulationConfig())
    brain.step(np.array([2.0, 0.0]))
    spikes = brain.step(np.array([0.0, 1.0]))
    assert not spikes[1]


def test_reject_invalid_input():
    brain = make_toy_brain()
    with pytest.raises(ValueError):
        brain.step(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        brain.step(reward=float("nan"))
