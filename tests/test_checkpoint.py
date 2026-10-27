import numpy as np
import pytest

from fruit_fly_stuff import load_checkpoint, save_checkpoint
from fruit_fly_stuff.demo import cue_response, make_toy_brain, train, experiment


def test_learning_persists_after_process_style_restore(tmp_path):
    path = tmp_path / "brain.npz"
    brain = make_toy_brain()
    train(brain)
    save_checkpoint(brain, path)
    restored = load_checkpoint(path)
    assert np.array_equal(restored.weights, brain.weights)
    assert restored.steps == brain.steps
    assert cue_response(restored)


def test_full_temporal_state_restored(tmp_path):
    brain = make_toy_brain()
    brain.step(np.array([2.0, 2.0, 0.0]))
    path = tmp_path / "brain.npz"
    save_checkpoint(brain, path)
    restored = load_checkpoint(path)
    assert np.array_equal(brain.previous_spikes, restored.previous_spikes)
    assert np.array_equal(brain.potential, restored.potential)
    assert np.array_equal(brain.eligibility, restored.eligibility)
    assert np.array_equal(brain.step(), restored.step())


def test_checkpoint_replacement(tmp_path):
    path = tmp_path / "brain.npz"
    brain = make_toy_brain()
    save_checkpoint(brain, path)
    train(brain)
    save_checkpoint(brain, path)
    assert load_checkpoint(path).weights[0] > 1.0


def test_corrupt_checkpoint_rejected(tmp_path):
    path = tmp_path / "brain.npz"
    np.savez(path, version=np.array([1]))
    with pytest.raises(ValueError, match="missing"):
        load_checkpoint(path)


def test_demo(tmp_path):
    result = experiment(tmp_path / "state" / "test.npz")
    assert result["before_training_cue_evokes_output"] is False
    assert result["after_restoring_cue_evokes_output"] is True
    assert (tmp_path / "state" / "test.npz").exists()
