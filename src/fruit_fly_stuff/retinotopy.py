"""Versioned FAFB v783 optic-column -> virtual hex-facet compiler.

Biological root IDs / column assignments are real. Their alignment to the
single-camera virtual facet mosaic is NOT calibrated. Require either an
explicit user calibration or --approximate; never silently imply biology.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np

from .whole_brain import load_graph, sha256_file

FACETS_PER_EYE = 721
FACETS = 1442
TYPES = ("L1", "L2")


def facet_coordinates():
    """Match engine/src/retina.rs axial-hex enumeration exactly."""
    coords = []
    for _side in range(2):
        for q in range(-15, 16):
            for r in range(max(-15, -q - 15), min(15, -q + 15) + 1):
                coords.append((q, r))
    a = np.asarray(coords, dtype=np.float32)
    if a.shape != (FACETS, 2):
        raise AssertionError("hex enumeration mismatch")
    return a


def read_calibration(file: Path):
    mapped = {}
    with file.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"hemisphere", "column_id", "facet_index"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("calibration needs hemisphere,column_id,facet_index")
        for row in reader:
            side = row["hemisphere"].lower().strip()
            ix = int(row["facet_index"])
            if side not in ("left", "right") or not 0 <= ix < FACETS:
                raise ValueError("invalid calibration hemisphere/index")
            if (ix < FACETS_PER_EYE) != (side == "left"):
                raise ValueError("calibration crosses hemisphere boundaries")
            key = (side, row["column_id"].strip())
            if key in mapped:
                raise ValueError("duplicate calibrated column")
            mapped[key] = ix
    if not mapped:
        raise ValueError("empty calibration file")
    return mapped


def compile_retinotopy(source: Path, cache_dir: Path, output: Path, *,
                      cell_types=TYPES, approximate=False, calibration=None):
    if (bool(approximate) + (calibration is not None)) != 1:
        raise ValueError("choose exactly one of --approximate and --calibration")
    _, roots, metadata = load_graph(cache_dir)
    if metadata.get("dataset") != "FlyWire FAFB v783":
        raise ValueError("expected FlyWire FAFB v783")
    rows = []
    with gzip.open(source, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"root_id", "hemisphere", "type", "column_id", "p", "q"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"unexpected FAFB column table: {reader.fieldnames}")
        for row in reader:
            if row["type"].strip() not in cell_types:
                continue
            side = row["hemisphere"].strip().lower()
            if side not in ("left", "right"):
                continue
            root = int(row["root_id"])
            idx = int(np.searchsorted(roots, np.uint64(root)))
            if idx >= len(roots) or int(roots[idx]) != root:
                continue
            rows.append((root, idx, side, row["type"].strip(),
                         row["column_id"].strip(), int(row["p"]), int(row["q"])))
    if not rows or any(not any(r[2] == side for r in rows) for side in ("left", "right")):
        raise ValueError("selected column types not found on both sides")
    mapping = read_calibration(calibration) if calibration else {}
    if approximate:
        facets = facet_coordinates()
        for side, offset in (("left", 0), ("right", FACETS_PER_EYE)):
            columns = sorted({(r[4], r[5], r[6]) for r in rows if r[2] == side})
            coords = np.array([(p, q) for _, p, q in columns], dtype=np.float32)
            centered = coords - coords.mean(axis=0)
            scale = np.max(np.abs(centered))
            if scale == 0:
                raise ValueError("insufficient columns for approximation")
            targets = centered * (15.0 / scale)
            delta = targets[:, None, :] - facets[offset:offset + FACETS_PER_EYE][None, :, :]
            distance = delta[..., 0]**2 + delta[..., 0]*delta[..., 1] + delta[..., 1]**2
            for (column, _, _), facet in zip(columns, distance.argmin(axis=1)):
                mapping[(side, column)] = int(offset + facet)
    entries = []
    seen = set()
    for root, idx, side, kind, col, p, q in sorted(rows):
        if (side, col) not in mapping or root in seen:
            continue
        seen.add(root)
        entries.append({"root_id": str(root), "neuron_index": idx,
                        "hemisphere": side, "type": kind, "column_id": col,
                        "p": p, "q": q, "facet_index": mapping[(side, col)]})
    if not entries:
        raise ValueError("calibration does not cover selected columns")
    result = {
        "schema": 1, "dataset": "FlyWire FAFB v783",
        "facet_layout": "two-721-projected-hex-halves-v2",
        "facets": FACETS,
        "mapping_kind": "UNVALIDATED_axial_fit" if approximate else "user_supplied_unverified_calibration",
        "source_column_sha256": sha256_file(source),
        "connectome_sha256": sha256_file(cache_dir / "adjacency.npz"),
        "calibration_sha256": sha256_file(calibration) if calibration else None,
        "neuron_count": len(roots), "selected_cell_types": sorted(cell_types),
        "mapped_neurons": len(entries), "unmapped_neurons": len(rows) - len(entries),
        "warning": "Actual FlyWire IDs/columns; facet geometry and neural coding are not scientifically calibrated.",
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    return {key: value for key, value in result.items() if key != "entries"}


def load_mapping(path: Path, cache_dir: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if (obj.get("schema") != 1 or obj.get("dataset") != "FlyWire FAFB v783" or
        obj.get("facet_layout") != "two-721-projected-hex-halves-v2" or
        obj.get("facets") != FACETS or
        obj.get("connectome_sha256") != sha256_file(cache_dir / "adjacency.npz")):
        raise ValueError("retina mapping incompatible with this graph/version")
    _, roots, _ = load_graph(cache_dir)
    source_file = cache_dir.parent / "column_assignment.csv.gz"
    if not source_file.is_file() or obj.get("source_column_sha256") != sha256_file(source_file):
        raise ValueError("retina source column checksum does not match the compiled mapping")
    rows = obj["entries"]
    if not rows:
        raise ValueError("empty retina mapping")
    indices = np.asarray([r["neuron_index"] for r in rows], dtype=np.int32)
    facets = np.asarray([r["facet_index"] for r in rows], dtype=np.int32)
    if (np.any(indices < 0) or np.any(indices >= len(roots)) or
        np.any(facets < 0) or np.any(facets >= FACETS) or
        any(str(int(roots[i])) != r["root_id"] for i, r in zip(indices, rows))):
        raise ValueError("retina manifest has neuron ID / facet mismatch")
    return indices, facets, obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/fafb783/column_assignment.csv.gz"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/fafb783/processed"))
    p.add_argument("--output", type=Path, default=Path("data/fafb783/processed/retinotopy.json"))
    p.add_argument("--types", default="L1,L2", help="columnar cell types (default L1,L2)")
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--approximate", action="store_true")
    modes.add_argument("--calibration", type=Path)
    args = p.parse_args()
    print(json.dumps(compile_retinotopy(args.source, args.cache_dir, args.output,
                       cell_types=tuple(s.strip() for s in args.types.split(",") if s.strip()),
                       approximate=args.approximate, calibration=args.calibration), indent=2))


if __name__ == "__main__":
    main()
