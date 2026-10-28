import io
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fruit_fly_stuff.neural_bridge import BrainAdapter, serve, strongest_disjoint_edges


def test_toy_sensor_spikes_drive_motor_after_one_step(tmp_path):
    adapter = BrainAdapter("toy", tmp_path, tmp_path / "state.npz")
    assert adapter.hello()["neurons"] == 4
    assert adapter.step(1, 1.0, 0.0)["left"] == 0
    second = adapter.step(2, 0.0, 0.0)
    assert second["left"] == 1
    assert second["right"] == 0


def test_checkpointed_bridge_recovers_temporal_state(tmp_path):
    ckpt = tmp_path / "brain.npz"
    first = BrainAdapter("toy", tmp_path, ckpt)
    first.step(1, 1.0, 0.0)
    first.save()
    other = BrainAdapter("toy", tmp_path, ckpt)
    assert other.hello()["steps"] == 1
    assert other.step(2, 0.0, 0.0)["left"] == 1


def test_toy_right_channel_is_independent(tmp_path):
    brain = BrainAdapter("toy", tmp_path, tmp_path / "brain.npz")
    brain.step(0, 0.0, 1.0)
    result = brain.step(1, 0.0, 0.0)
    assert result["left"] == 0 and result["right"] == 1


def test_strongest_two_nonoverlapping_connections():
    g = csr_matrix(np.array([
        [0, 0, 0, 0],
        [0.9, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0.8, 0],
    ], dtype=np.float32))
    assert strongest_disjoint_edges(g) == ((0, 1), (2, 3))


def test_serve_protocol_and_clean_exit(tmp_path, monkeypatch):
    lines = [
        {"type": "step", "version": 1, "seq": 1, "left": 1.0, "right": 0.0},
        {"type": "step", "version": 1, "seq": 2, "left": 0.0, "right": 0.0},
        {"type": "close", "version": 1},
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps(x) + "\n" for x in lines)))
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    ckpt = tmp_path / "persistent.npz"
    assert serve("toy", tmp_path, ckpt, save_every=3) == 0
    parsed = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [x["type"] for x in parsed] == ["ready", "motor", "motor", "saved"]
    assert parsed[2]["left"] == 1
    assert ckpt.exists()


def test_bad_sensor_range_rejected(tmp_path):
    b = BrainAdapter("toy", tmp_path, tmp_path / "memory.npz")
    with pytest.raises(ValueError):
        b.step(1, -1.0, 0.0)
