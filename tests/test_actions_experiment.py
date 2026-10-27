import pytest

from fruit_fly_stuff.actions_experiment import render_summary, run_experiment
from fruit_fly_stuff import load_checkpoint


def test_learning_across_two_independent_action_runs(tmp_path):
    state = tmp_path / "toy-brain.npz"
    first = run_experiment(state, trials=3)
    assert first["loaded_previous_memory"] is False
    assert first["cue_weight_after"] == pytest.approx(0.64)
    assert first["cue_response_after_restore"] is False

    # New invocation: previous in-memory brain object is gone.
    second = run_experiment(state, trials=3)
    assert second["loaded_previous_memory"] is True
    assert second["cue_weight_before"] == first["cue_weight_after"]
    assert second["cue_weight_after"] == pytest.approx(1.18)
    assert second["cue_response_before"] is False
    assert second["cue_response_after_restore"] is True
    assert second["steps_before"] == first["steps_after"]
    assert load_checkpoint(state).steps == second["steps_after"]
    assert "Previous saved memory restored" in render_summary(second)


def test_explicit_reset_restarts_the_weights(tmp_path):
    state = tmp_path / "toy-brain.npz"
    run_experiment(state, trials=6)
    reset = run_experiment(state, trials=1, reset=True)
    assert reset["loaded_previous_memory"] is False
    assert reset["reset_requested"] is True
    assert reset["cue_weight_before"] == pytest.approx(0.1)
    assert reset["cue_weight_after"] == pytest.approx(0.28)


def test_bad_trials_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="between"):
        run_experiment(tmp_path / "toy.npz", trials=0)
