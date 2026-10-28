import csv
import gzip

import numpy as np
import pytest

from fruit_fly_stuff.whole_brain import WholeBrain, load_graph, prepare, sha256_file


def write_gzip_csv(path, names, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(names)
        writer.writerows(rows)


def sample_data(tmp_path):
    write_gzip_csv(tmp_path / "neurons.csv.gz", ["root_id", "nt_type"],
                   [[44, "ACH"], [11, "ACH"], [33, "GABA"], [22, "ACH"]])
    write_gzip_csv(tmp_path / "connections_no_threshold.csv.gz",
                   ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
                   [[11, 22, "A", 2, "ACH"], [11, 22, "B", 3, "ACH"],
                    [22, 33, "A", 4, "ACH"], [22, 44, "A", 1, "ACH"]])
    return tmp_path


def test_import_preserves_every_root_and_aggregates_regions(tmp_path):
    location = sample_data(tmp_path)
    info = prepare(location)
    graph, ids, loaded = load_graph(location / "processed")
    assert info["neurons"] == 4
    assert info["directed_neuron_pairs"] == 3
    assert info["input_connection_rows"] == 4
    assert info["synaptic_contacts_in_input"] == 10
    assert ids.tolist() == [11, 22, 33, 44]
    assert graph.shape == (4, 4)
    assert graph[1, 0] == pytest.approx(1)
    assert graph[2, 1] == pytest.approx(1)
    assert graph[3, 1] == pytest.approx(1)
    assert graph[0].nnz == 0
    assert loaded["neurons_sha256"] == sha256_file(location / "neurons.csv.gz")


def test_full_graph_activity_and_persistence(tmp_path):
    location = sample_data(tmp_path)
    prepare(location)
    graph, ids, _ = load_graph(location / "processed")
    brain = WholeBrain(graph, ids, gain=2.0, leak=0.8)
    assert brain.step(np.array([0], dtype=np.int32)) == 1
    assert brain.step(np.array([], dtype=np.int32)) == 1
    checkpoint = location / "memory.npz"
    checksum = sha256_file(location / "processed" / "adjacency.npz")
    brain.save(checkpoint, graph_sha=checksum)
    restored = WholeBrain(graph, ids, gain=2.0, leak=0.8)
    restored.restore(checkpoint, graph_sha=checksum)
    assert restored.steps == 2
    assert np.array_equal(restored.voltage, brain.voltage)
    assert np.array_equal(restored.spikes, brain.spikes)
    assert restored.step(np.array([], dtype=np.int32)) == brain.step(np.array([], dtype=np.int32))


def test_wrong_graph_rejected(tmp_path):
    location = sample_data(tmp_path)
    prepare(location)
    graph, ids, _ = load_graph(location / "processed")
    brain = WholeBrain(graph, ids)
    brain.save(location / "memory.npz", graph_sha="abc")
    with pytest.raises(ValueError, match="incompatible"):
        brain.restore(location / "memory.npz", graph_sha="different")


def test_unfiltered_source_is_explicit(tmp_path):
    location = sample_data(tmp_path)
    (location / "connections_no_threshold.csv.gz").rename(location / "connections.csv.gz")
    with pytest.raises(FileNotFoundError):
        prepare(location)
    assert prepare(location, variant="filtered")["variant"] == "filtered"


def test_duplicate_neuron_roots_rejected(tmp_path):
    sample_data(tmp_path)
    write_gzip_csv(tmp_path / "neurons.csv.gz", ["root_id"], [[11], [11]])
    with pytest.raises(ValueError, match="duplicate"):
        prepare(tmp_path)


def test_checkpoint_rejects_incompatible_sensory_mapping(tmp_path):
    location = sample_data(tmp_path)
    prepare(location)
    graph, ids, _ = load_graph(location / "processed")
    brain = WholeBrain(graph, ids)
    checkpoint = location / "mapping-memory.npz"
    brain.step(np.array([0], dtype=np.int32))
    brain.save(checkpoint, graph_sha="graph-123", context_sha="retina-layout-A")
    restored = WholeBrain(graph, ids)
    with pytest.raises(ValueError, match="incompatible"):
        restored.restore(checkpoint, graph_sha="graph-123",
                         context_sha="retina-layout-B")
    with pytest.raises(ValueError, match="incompatible"):
        restored.restore(checkpoint, graph_sha="graph-123")
    restored.restore(checkpoint, graph_sha="graph-123",
                     context_sha="retina-layout-A")
    assert restored.steps == 1
