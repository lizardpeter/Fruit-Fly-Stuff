import csv
import gzip
import json

import numpy as np
import pytest

from fruit_fly_stuff.retinotopy import facet_coordinates, compile_retinotopy, load_mapping
from fruit_fly_stuff.whole_brain import prepare


def fixture(tmp_path):
    with gzip.open(tmp_path / "neurons.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_id"])
        w.writerows([[100], [200], [300], [400], [500]])
    with gzip.open(tmp_path / "connections_no_threshold.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pre_root_id", "post_root_id", "syn_count"])
        w.writerows([[100, 300, 2], [200, 400, 3], [300, 500, 1]])
    prepare(tmp_path)
    source = tmp_path / "column_assignment.csv.gz"
    with gzip.open(source, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root_id", "hemisphere", "type", "column_id", "p", "q"])
        w.writerows([
            [100, "left", "L1", "a", -2, 0],
            [200, "right", "L2", "b", 2, 0],
            [300, "left", "L1", "c", -1, 1],
            [400, "right", "L2", "d", 1, -1],
            [500, "right", "T4a", "e", 0, 0],
        ])
    return source, tmp_path / "processed"


def test_hex_layout_corresponds_to_rust():
    coords = facet_coordinates()
    assert coords.shape == (1442, 2)
    assert tuple(coords[0]) == (-15, 0)
    assert tuple(coords[720]) == (15, 0)
    assert tuple(coords[721]) == (-15, 0)


def test_approximate_mapping_is_flagged_and_root_ids_match(tmp_path):
    source, cache = fixture(tmp_path)
    output = tmp_path / "retinotopy.json"
    meta = compile_retinotopy(source, cache, output, approximate=True)
    assert meta["mapping_kind"] == "UNVALIDATED_axial_fit"
    neuron_indices, facets, manifest = load_mapping(output, cache)
    assert len(neuron_indices) == 4
    assert len(facets) == 4
    for row in manifest["entries"]:
        assert (row["hemisphere"] == "left") == (row["facet_index"] < 721)
    assert set(manifest["selected_cell_types"]) == {"L1", "L2"}


def test_calibration_required(tmp_path):
    source, cache = fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        compile_retinotopy(source, cache, tmp_path / "out.json")


def test_user_calibration_is_versioned(tmp_path):
    source, cache = fixture(tmp_path)
    calib = tmp_path / "calibration.csv"
    calib.write_text("hemisphere,column_id,facet_index\nleft,a,12\nright,b,735\n")
    path = tmp_path / "out.json"
    meta = compile_retinotopy(source, cache, path, calibration=calib)
    assert meta["mapped_neurons"] == 2
    _, facets, obj = load_mapping(path, cache)
    assert set(facets.tolist()) == {12, 735}
    assert obj["calibration_sha256"]


def test_crossed_eye_calibration_rejected(tmp_path):
    source, cache = fixture(tmp_path)
    calib = tmp_path / "calibration.csv"
    calib.write_text("hemisphere,column_id,facet_index\nleft,a,830\n")
    with pytest.raises(ValueError, match="crosses hemisphere"):
        compile_retinotopy(source, cache, tmp_path / "out.json", calibration=calib)


def test_wrong_graph_rejected(tmp_path):
    source, cache = fixture(tmp_path)
    path = tmp_path / "out.json"
    compile_retinotopy(source, cache, path, approximate=True)
    data = json.loads(path.read_text())
    data["connectome_sha256"] = "not this connectome"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incompatible"):
        load_mapping(path, cache)
