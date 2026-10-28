import io
import json
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from fruit_fly_stuff.neural_bridge import BrainAdapter, serve, strongest_disjoint_edges, route_visual_channels


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


def test_weak_rendered_visual_feature_drives_toy_neurons(tmp_path):
    brain = BrainAdapter("toy", tmp_path, tmp_path / "vision.npz")
    first = brain.step(0, 0.062, 0.0)
    second = brain.step(1, 0.0, 0.0)
    assert first["left"] == 0
    assert second["left"] == 1
    assert second["right"] == 0


def test_retina_on_off_pathway_routing_is_type_specific():
    neuron_ids = np.array([12, 23, 34, 45], dtype=np.int32)
    facets = np.array([5, 5, 730, 730], dtype=np.int32)
    kinds = np.array(["L1", "L2", "L1", "L2"])
    on = np.zeros(1442, dtype=np.float32)
    off = np.zeros(1442, dtype=np.float32)
    on[5], off[730] = 0.8, 0.9
    result = route_visual_channels(neuron_ids, facets, kinds, on, off)
    assert result.tolist() == [12, 45]
    on[5], off[730] = 0.0, 0.0
    assert route_visual_channels(neuron_ids, facets, kinds, on, off).size == 0


@pytest.mark.parametrize("invalid", [
    [0.0] * 1441, [0.0] * 1442 + [0.1],
    [-1.0] + [0.0] * 1441, [float("nan")] + [0.0] * 1441,
])
def test_retina_rejects_incorrect_or_nonfinite_signals(invalid):
    with pytest.raises(ValueError):
        route_visual_channels(
            np.array([2]), np.array([0]), np.array(["L1"]),
            invalid, np.zeros(1442, dtype=np.float32),
        )


def test_retina_requires_both_polarities_and_supported_cell_types():
    retina = np.zeros(1442, dtype=np.float32)
    with pytest.raises(ValueError, match="both ON and OFF"):
        route_visual_channels([1], [1], ["L1"], retina, None)
    with pytest.raises(ValueError, match="only mapped L1 and L2"):
        route_visual_channels([1], [1], ["T4a"], retina, retina)


def test_l2_off_channel_independent_of_l1_on():
    on = np.zeros(1442, dtype=np.float32)
    off = np.zeros(1442, dtype=np.float32)
    off[12] = 0.6
    assert route_visual_channels([9, 10], [12, 12], ["L1", "L2"], on, off).tolist() == [10]
