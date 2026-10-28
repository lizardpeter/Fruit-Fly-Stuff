"""Build a typed visual/descending neuron mapping from FlyWire v783 labels.

Source: FlyWire Codex FAFB v783 classification.csv.gz (Schlegel et al.;
Matsliah et al. optic-lobe annotations). The *identities* are from biology;
the mapping of camera features to those neurons, and spikes to movements, is
NOT established by biological data.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

from .whole_brain import sha256_file


def classified_groups(path: Path, root_ids: np.ndarray, *,
                      max_visual_per_side: int = 256) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download FAFB v783 classification.csv.gz "
            "with 'python -m fruit_fly_stuff.whole_brain download --annotations'."
        )
    chosen: dict[str, list[int]] = {
        "visual_left": [], "visual_right": [],
        "descending_left": [], "descending_right": [],
    }
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"root_id", "super_class", "side"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"classification table must contain {sorted(required)}")
        for line in reader:
            root = int(line["root_id"])
            if root in seen:
                raise ValueError(f"duplicated root id in classification file: {root}")
            seen.add(root)
            side = line["side"].lower().strip()
            if side not in ("left", "right"):
                continue
            group = line["super_class"].lower().strip()
            if group == "optic":
                key = f"visual_{side}"
            elif group == "descending":
                key = f"descending_{side}"
            else:
                continue
            pos = int(np.searchsorted(root_ids, np.uint64(root)))
            if pos >= len(root_ids) or int(root_ids[pos]) != root:
                continue
            chosen[key].append(pos)
    # Sort by original root ID, not CSV row order, for stable reproducibility.
    groups = {}
    for key, indices in chosen.items():
        indices = sorted(indices, key=lambda i: int(root_ids[i]))
        if key.startswith("visual"):
            indices = indices[:max_visual_per_side]
        if not indices:
            raise ValueError(f"no {key} cells found in FAFB classifications")
        groups[key] = np.asarray(indices, dtype=np.int32)
    groups["metadata"] = {
        "source": "FlyWire FAFB v783 classification.csv.gz",
        "classification_sha256": sha256_file(path),
        "visual_super_class": "optic",
        "motor_super_class": "descending",
        "visual_selection": "first 256 by root ID in each side (artificial stimulation)",
        "motor_readout": "any detected descending neuron spike per side (artificial movement decoder)",
        "counts": {key: int(len(value)) for key, value in groups.items()},
    }
    return groups
