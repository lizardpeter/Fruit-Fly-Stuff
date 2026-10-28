import csv
import gzip

import numpy as np
import pytest

from fruit_fly_stuff.annotated_mapping import classified_groups


def test_classified_vision_and_motor_indices(tmp_path):
    path = tmp_path / "classification.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("root_id", "super_class", "side"))
        writer.writeheader()
        writer.writerows([
            {"root_id": 22, "super_class": "optic", "side": "left"},
            {"root_id": 44, "super_class": "optic", "side": "right"},
            {"root_id": 11, "super_class": "descending", "side": "left"},
            {"root_id": 33, "super_class": "descending", "side": "right"},
        ])
    ids = np.array([11, 22, 33, 44], dtype=np.uint64)
    mapping = classified_groups(path, ids)
    assert mapping["visual_left"].tolist() == [1]
    assert mapping["visual_right"].tolist() == [3]
    assert mapping["descending_left"].tolist() == [0]
    assert mapping["descending_right"].tolist() == [2]
    assert len(mapping["metadata"]["classification_sha256"]) == 64


def test_missing_groups_fails_clearly(tmp_path):
    path = tmp_path / "classification.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["root_id", "super_class", "side"])
        writer.writerow([11, "optic", "left"])
    with pytest.raises(ValueError, match="no .* cells"):
        classified_groups(path, np.array([11, 22], dtype=np.uint64))
