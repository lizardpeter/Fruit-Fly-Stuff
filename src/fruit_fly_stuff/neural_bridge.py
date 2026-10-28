"""Versioned local JSON-lines bridge between the Rust world and Python neural engine.

The default toy network is 4 *synthetic* neurons; full mode loads the actual
FlyWire FAFB v783 graph, but uses a clearly labeled *topological probe* rather
than a biologically established sensory/motor map.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .annotated_mapping import classified_groups
from .simulation import Connectome, SpikingBrain
from .whole_brain import WholeBrain, load_graph, sha256_file

PROTOCOL_VERSION = 1


def toy_brain() -> SpikingBrain:
    return SpikingBrain(Connectome(
        n_neurons=4,
        source=np.array([0, 1], dtype=np.int64),
        target=np.array([2, 3], dtype=np.int64),
        weight=np.array([1.25, 1.25], dtype=np.float64),
        plastic=np.array([False, False], dtype=bool),
    ))


def strongest_disjoint_edges(graph) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select two distinct high-weight edges for *software validation only*.

    This is NOT identification of photoreceptors, descending motor neurons,
    or any other anatomically grounded cell type.
    """
    if graph.nnz < 2:
        raise ValueError("need at least two directed connections to build a dual-channel probe")
    # Top weights are likely to drive downstream neurons under the current
    # artificial point-neuron gain; source/target IDs are retained in metadata.
    n_candidates = min(5000, graph.nnz)
    top = np.argpartition(graph.data, -n_candidates)[-n_candidates:]
    top = top[np.lexsort((top, -graph.data[top]))]
    row_for_entry = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    chosen: list[tuple[int, int]] = []
    used = set()
    for entry in top:
        post = int(row_for_entry[entry])
        pre = int(graph.indices[entry])
        if pre in used or post in used or pre == post:
            continue
        chosen.append((pre, post))
        used.update([pre, post])
        if len(chosen) == 2:
            return chosen[0], chosen[1]
    raise ValueError("no two disjoint directed connections in the candidate set")


class BrainAdapter:
    def __init__(self, mode: str, cache_dir: Path, checkpoint: Path, mapping: str = "classified"):
        self.mode = mode
        self.checkpoint = checkpoint
        if mapping not in ("classified", "probes"):
            raise ValueError("mapping must be classified or probes")
        self.mapping_name = mapping
        self.groups = None
        if mode == "toy":
            self.brain = load_checkpoint(checkpoint) if checkpoint.exists() else toy_brain()
            if self.brain.connectome.n_neurons != 4:
                raise ValueError("toy checkpoint must contain the 4-neuron bridge model")
            self.left = (0, 2)
            self.right = (1, 3)
            self.graph_hash = None
            self.edges = 2
            self.ids = None
        elif mode == "full":
            graph, ids, metadata = load_graph(cache_dir)
            self.graph_hash = sha256_file(cache_dir / "adjacency.npz")
            self.brain = WholeBrain(graph, ids)
            if checkpoint.exists():
                self.brain.restore(checkpoint, graph_sha=self.graph_hash)
            if mapping == "classified":
                self.groups = classified_groups(cache_dir.parent / "classification.csv.gz", ids)
                self.left = self.right = None
            else:
                self.left, self.right = strongest_disjoint_edges(graph)
            self.edges = graph.nnz
            self.ids = ids
            self.metadata = metadata
        else:
            raise ValueError("mode must be toy or full")

    def hello(self) -> dict:
        def link(pair):
            if pair is None:
                return None
            if self.ids is None:
                return {"input_index": pair[0], "output_index": pair[1]}
            return {
                "input_root_id": str(int(self.ids[pair[0]])),
                "output_root_id": str(int(self.ids[pair[1]])),
            }
        return {
            "type": "ready", "version": PROTOCOL_VERSION, "mode": self.mode,
            "neurons": int(len(self.ids)) if self.ids is not None else 4,
            "directed_pairs": int(self.edges),
            "steps": int(self.brain.steps),
            "left_probe": link(self.left),
            "right_probe": link(self.right),
            "mapping": "synthetic toy circuit" if self.mode == "toy" else
                ("classified optic-lobe inputs and descending-neuron readout; "
                 "pixel-to-neuron and motor mappings are NOT biologically calibrated"
                 if self.groups is not None else
                 "strongest disjoint FlyWire neuron-pair probes (not anatomical motor/sensory assignments)"),
            "anatomical_groups": self.groups["metadata"] if self.groups is not None else None,
        }

    def step(self, sequence: int, left: float, right: float) -> dict:
        for x in (left, right):
            if not np.isfinite(x) or x < 0 or x > 1:
                raise ValueError("left and right must be finite in [0, 1]")
        if self.groups is not None:
            # Anatomically assigned *broad classes*, experimental encoding:
            # strong camera signal drives up to 256 optic-lobe neurons by side.
            excite_parts = []
            if left >= 0.04:
                excite_parts.append(self.groups["visual_left"])
            if right >= 0.04:
                excite_parts.append(self.groups["visual_right"])
            excite = (np.concatenate(excite_parts) if excite_parts else
                      np.empty(0, dtype=np.int32))
            self.brain.step(excite)
            spikes = self.brain.spikes
            # Descending neurons are real classifications. We do not know
            # whether their firing means left/right walking in this model.
            left_spike = bool(np.any(spikes[self.groups["descending_left"]]))
            right_spike = bool(np.any(spikes[self.groups["descending_right"]]))
        else:
            excite = np.array([idx for input_value, (idx, _) in
                               [(left, self.left), (right, self.right)]
                               if input_value >= 0.04], dtype=np.int32)
            if self.mode == "toy":
                drive = np.zeros(4, dtype=np.float64)
                drive[excite] = 1.2
                spikes = self.brain.step(drive)
            else:
                self.brain.step(excite)
                spikes = self.brain.spikes
            left_spike = bool(spikes[self.left[1]])
            right_spike = bool(spikes[self.right[1]])
        # Only measured spikes from the network drive outputs. Smoothing and
        # robot-like motor translation happen in the Rust environment.
        return {
            "type": "motor", "version": PROTOCOL_VERSION,
            "seq": sequence, "steps": int(self.brain.steps),
            "left": int(left_spike), "right": int(right_spike),
            "active": int(np.count_nonzero(spikes)),
        }

    def save(self) -> None:
        if self.mode == "full":
            self.brain.save(self.checkpoint, graph_sha=self.graph_hash)
        else:
            save_checkpoint(self.brain, self.checkpoint)


def serve(mode: str, cache_dir: Path, checkpoint: Path, save_every: int = 300, mapping: str = "classified") -> int:
    adapter = BrainAdapter(mode, cache_dir, checkpoint, mapping=mapping)
    sys.stdout.write(json.dumps(adapter.hello()) + "\n")
    sys.stdout.flush()
    previous_seq = -1
    try:
        for raw in sys.stdin:
            try:
                if len(raw) > 65536:
                    raise ValueError("request too long")
                msg = json.loads(raw)
                if msg.get("version") != PROTOCOL_VERSION:
                    raise ValueError("unsupported protocol version")
                if msg.get("type") == "close":
                    adapter.save()
                    print(json.dumps({"type": "saved", "steps": adapter.brain.steps}), flush=True)
                    break
                if msg.get("type") != "step":
                    raise ValueError("unknown request type")
                seq = msg["seq"]
                if not isinstance(seq, int) or isinstance(seq, bool) or seq <= previous_seq:
                    raise ValueError("sequence must be a strictly increasing integer")
                left = float(msg["left"])
                right = float(msg["right"])
                answer = adapter.step(seq, left, right)
                previous_seq = seq
                print(json.dumps(answer), flush=True)
                if adapter.brain.steps % save_every == 0:
                    adapter.save()
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                print(json.dumps({"type": "error", "message": str(exc)}), flush=True)
    finally:
        adapter.save()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("toy", "full"), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/fafb783/processed"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--mapping", choices=("classified", "probes"), default="classified")
    args = parser.parse_args()
    if args.save_every < 1:
        parser.error("--save-every must be >= 1")
    checkpoint = args.checkpoint or Path(f"state/bridge-{args.mode}.npz")
    raise SystemExit(serve(args.mode, args.cache_dir, checkpoint, args.save_every, args.mapping))


if __name__ == "__main__":
    main()
