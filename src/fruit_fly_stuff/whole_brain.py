"""FAFB v783 sparse *mathematical* whole-connectome loader and runner.

Actual neuron IDs and measured directed contact counts come from FlyWire Codex.
Membrane dynamics, normalization, gains and stimulation are UNVALIDATED toy
assumptions, not a published physiological model of the fruit fly.
"""
from __future__ import annotations

import argparse
from array import array
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz

DATA_BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data/fafb/783"
DATASET = "FlyWire FAFB v783"
SCHEMA = 1
FILES = {
    "filtered": "connections.csv.gz",
    "unfiltered": "connections_no_threshold.csv.gz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_data(data_dir: Path, *, variant: str = "unfiltered", annotations: bool = False, retinotopy: bool = False) -> None:
    """Fetch official public static exports (no credentials assumed).

    If mirrors or access policies change, users can manually place the two
    files in data_dir from the Codex Download Data page instead.
    """
    if variant not in FILES:
        raise ValueError("variant must be filtered or unfiltered")
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("neurons.csv.gz", FILES[variant], *(("classification.csv.gz",) if annotations else ()), *(("column_assignment.csv.gz",) if retinotopy else ())):
        destination = data_dir / name
        if destination.exists() and destination.stat().st_size > 0:
            print(f"Already downloaded: {destination}")
            continue
        url = f"{DATA_BASE_URL}/{name}"
        print(f"Downloading {url}", flush=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            with urlopen(url, timeout=60) as response, partial.open("wb") as output:
                while block := response.read(1 << 20):
                    output.write(block)
            if partial.stat().st_size == 0:
                raise ValueError("empty dataset response")
            partial.replace(destination)
        except (HTTPError, URLError, OSError, ValueError) as error:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {name}: {error}. Get it through "
                "https://codex.flywire.ai/api/download (FAFB v783) "
                f"and place it in {data_dir}."
            ) from error


def _read_roots(path: Path) -> np.ndarray:
    roots = array("Q")
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or "root_id" not in reader.fieldnames:
            raise ValueError("neurons file needs root_id column")
        for row in reader:
            roots.append(int(row["root_id"]))
    if not roots:
        raise ValueError("neurons file is empty")
    ids = np.array(roots, dtype=np.uint64)
    ids.sort()
    if np.any(ids[1:] == ids[:-1]):
        raise ValueError("duplicate root_id values in neuron table")
    return ids


def prepare(data_dir: Path, *, variant: str = "unfiltered",
            cache_dir: Path | None = None) -> dict:
    """Convert every available edge row to a summed sparse pair adjacency.

    A column-major edge list is retained in compact typed arrays before SciPy
    coalesces duplicate (pre,post) pairs across neuropils. All neuron IDs,
    including those without retained edges, remain in the network.
    """
    if variant not in FILES:
        raise ValueError("variant must be filtered or unfiltered")
    neurons_file = data_dir / "neurons.csv.gz"
    edges_file = data_dir / FILES[variant]
    for file in (neurons_file, edges_file):
        if not file.is_file():
            raise FileNotFoundError(
                f"Missing {file}. Run 'download' or use Codex Download Data "
                "with the same FAFB v783 snapshot."
            )
    root_ids = _read_roots(neurons_file)
    before = array("Q")
    after = array("Q")
    count = array("f")
    contacts = 0
    with gzip.open(edges_file, "rt", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        needed = {"pre_root_id", "post_root_id", "syn_count"}
        if not reader.fieldnames or not needed.issubset(reader.fieldnames):
            raise ValueError(f"connections file needs columns: {sorted(needed)}")
        for row in reader:
            syn = int(row["syn_count"])
            if syn <= 0:
                raise ValueError("synapse count must be positive")
            before.append(int(row["pre_root_id"]))
            after.append(int(row["post_root_id"]))
            count.append(float(syn))
            contacts += syn
    # Searchsorted preserves full 64-bit root IDs; never cast IDs to floats.
    sources = np.frombuffer(before, dtype=np.uint64)
    targets = np.frombuffer(after, dtype=np.uint64)
    weights = np.frombuffer(count, dtype=np.float32)
    pre = np.searchsorted(root_ids, sources)
    post = np.searchsorted(root_ids, targets)
    n = root_ids.size
    present = (
        (pre < n) & (post < n) &
        (root_ids[np.minimum(pre, n - 1)] == sources) &
        (root_ids[np.minimum(post, n - 1)] == targets)
    )
    dropped = int(np.count_nonzero(~present))
    if dropped:
        print(f"Warning: skipped {dropped} rows referencing absent neuron IDs")
    # matrix[post,pre] equals incoming contact count from presynaptic pre.
    graph = coo_matrix((weights[present],
                       (post[present].astype(np.int32),
                        pre[present].astype(np.int32))),
                       shape=(n, n), dtype=np.float32).tocsr()
    graph.sum_duplicates()
    graph.sort_indices()
    raw_pairs = int(graph.nnz)
    # Illustrative normalization: each target receives total weight at most 1.
    incoming = np.asarray(graph.sum(axis=1)).ravel()
    inv = 1.0 / np.maximum(incoming, 1.0)
    graph.data *= np.repeat(inv, np.diff(graph.indptr)).astype(np.float32)
    if not np.all(np.isfinite(graph.data)):
        raise ValueError("invalid weight values in imported graph")
    cache_dir = Path(cache_dir or data_dir / "processed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_npz(cache_dir / "adjacency.npz", graph, compressed=True)
    np.save(cache_dir / "root_ids.npy", root_ids, allow_pickle=False)
    info = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "variant": variant,
        "neurons": int(n),
        "directed_neuron_pairs": raw_pairs,
        "input_connection_rows": len(before),
        "skipped_rows_unknown_neuron": dropped,
        "synaptic_contacts_in_input": contacts,
        "weights": "incoming-contact-fraction (illustrative, all nonnegative)",
        "neurons_file": neurons_file.name,
        "neurons_sha256": sha256_file(neurons_file),
        "connections_file": edges_file.name,
        "connections_sha256": sha256_file(edges_file),
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


def load_graph(cache_dir: Path) -> tuple[csr_matrix, np.ndarray, dict]:
    info = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if info.get("schema") != SCHEMA or info.get("dataset") != DATASET:
        raise ValueError("unsupported processed connectome schema/dataset")
    graph = load_npz(cache_dir / "adjacency.npz").tocsr()
    ids = np.load(cache_dir / "root_ids.npy", allow_pickle=False)
    if (ids.ndim != 1 or ids.dtype != np.uint64 or
        graph.shape != (len(ids), len(ids)) or
        len(ids) != info["neurons"] or graph.nnz != info["directed_neuron_pairs"]):
        raise ValueError("inconsistent processed connectome files")
    return graph, ids, info


class WholeBrain:
    """Toy point-neuron recurrent simulation using the *real* adjacency graph.

    No plasticity, biological calibration, vision or motor interface is
    included here. Checkpoints preserve voltage and spikes across restarts.
    """

    def __init__(self, graph: csr_matrix, ids: np.ndarray, *,
                 gain: float = 3.0, leak: float = 0.85):
        if graph.shape != (len(ids), len(ids)) or len(ids) == 0:
            raise ValueError("graph shape must match neuron IDs")
        if not (np.isfinite(gain) and gain >= 0 and np.isfinite(leak) and 0 <= leak < 1):
            raise ValueError("gain must be nonnegative, leak must lie in [0,1)")
        self.graph = graph.astype(np.float32, copy=False)
        self.ids = ids
        self.gain = float(gain)
        self.leak = float(leak)
        self.voltage = np.zeros(len(ids), dtype=np.float32)
        self.spikes = np.zeros(len(ids), dtype=bool)
        self.steps = 0

    def step(self, stimulated: np.ndarray) -> int:
        """One synchronous discrete step; stimulus IDs are index positions."""
        if stimulated.ndim != 1 or stimulated.dtype.kind not in "iu":
            raise ValueError("stimulated must be a 1-D integer index array")
        if len(stimulated) and (stimulated.min() < 0 or stimulated.max() >= len(self.ids)):
            raise ValueError("stimulated indices outside network")
        currents = self.graph @ self.spikes.astype(np.float32)
        self.voltage = self.leak * self.voltage + self.gain * currents
        self.voltage[stimulated] += 1.1
        self.spikes = self.voltage >= 1.0
        self.voltage[self.spikes] = 0.0
        self.steps += 1
        return int(np.count_nonzero(self.spikes))

    def save(self, path: Path, *, graph_sha: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".brain-",
                                             suffix=".npz", delete=False) as stream:
                temp_path = Path(stream.name)
                np.savez_compressed(
                    stream, version=np.array([1], dtype=np.int32),
                    steps=np.array([self.steps], dtype=np.int64),
                    gain=np.array([self.gain]), leak=np.array([self.leak]),
                    voltage=self.voltage, spikes=self.spikes,
                    graph_sha=np.array([graph_sha]),
                )
                stream.flush()
                os.fsync(stream.fileno())
            temp_path.replace(path)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def restore(self, path: Path, *, graph_sha: str) -> None:
        with np.load(path, allow_pickle=False) as state:
            if (int(state["version"][0]) != 1 or
                str(state["graph_sha"][0]) != graph_sha or
                state["voltage"].shape != self.voltage.shape or
                state["spikes"].shape != self.spikes.shape or
                state["spikes"].dtype.kind != "b" or
                float(state["gain"][0]) != self.gain or
                float(state["leak"][0]) != self.leak):
                raise ValueError("checkpoint is incompatible with this graph/configuration")
            volts = state["voltage"].astype(np.float32)
            if not np.all(np.isfinite(volts)):
                raise ValueError("checkpoint contains non-finite voltage")
            steps = int(state["steps"][0])
            if steps < 0:
                raise ValueError("checkpoint has invalid step count")
            self.voltage[:] = volts
            self.spikes[:] = state["spikes"]
            self.steps = steps


def run(cache_dir: Path, checkpoint: Path, *, steps: int = 100,
        stimulated_count: int = 100, gain: float = 3.0, leak: float = 0.85,
        forever: bool = False, tick_seconds: float = 0.0,
        checkpoint_every: int = 500) -> None:
    graph, ids, info = load_graph(cache_dir)
    # Hash the cache, not a filename, to prevent resuming the wrong wiring.
    graph_sha = sha256_file(cache_dir / "adjacency.npz")
    brain = WholeBrain(graph, ids, gain=gain, leak=leak)
    if checkpoint.exists():
        brain.restore(checkpoint, graph_sha=graph_sha)
        print(f"Resumed saved state at step {brain.steps}", flush=True)
    if stimulated_count < 0 or stimulated_count > len(ids) or steps < 1:
        raise ValueError("invalid stimulated_count or steps")
    if checkpoint_every < 1 or tick_seconds < 0:
        raise ValueError("checkpoint_every must be positive; tick_seconds >=0")
    print(f"{info['dataset']}: {len(ids):,} neurons, "
          f"{graph.nnz:,} directed pairs, {info['synaptic_contacts_in_input']:,} "
          f"input synaptic contacts, variant={info['variant']}", flush=True)
    print("This is a simplified mathematical graph simulation, not a biological fly.", flush=True)
    stimulate = np.arange(stimulated_count, dtype=np.int32)
    done = 0
    try:
        while forever or done < steps:
            t0 = time.monotonic()
            active = brain.step(stimulate)
            done += 1
            if done % 25 == 0 or done == 1:
                print(f"step={brain.steps} firing={active} / {len(ids)}", flush=True)
            if done % checkpoint_every == 0:
                brain.save(checkpoint, graph_sha=graph_sha)
                print(f"Saved {checkpoint}", flush=True)
            sleep = tick_seconds - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("Stopping and saving state.", flush=True)
    finally:
        brain.save(checkpoint, graph_sha=graph_sha)
        print(f"Checkpoint: {checkpoint} (step {brain.steps})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for cmd in ("download", "prepare"):
        sub = commands.add_parser(cmd)
        sub.add_argument("--data-dir", type=Path, default=Path("data/fafb783"))
        sub.add_argument("--variant", choices=tuple(FILES), default="unfiltered")
        if cmd == "prepare":
            sub.add_argument("--cache-dir", type=Path)
        else:
            sub.add_argument("--annotations", action="store_true",
                             help="also download source-matched optic/descending classifications")
            sub.add_argument("--retinotopy", action="store_true",
                             help="download source-matched visual column assignments")
    sub = commands.add_parser("run")
    sub.add_argument("--cache-dir", type=Path, default=Path("data/fafb783/processed"))
    sub.add_argument("--checkpoint", type=Path, default=Path("state/whole-brain.npz"))
    sub.add_argument("--steps", type=int, default=100)
    sub.add_argument("--stimulated-count", type=int, default=100)
    sub.add_argument("--gain", type=float, default=3.0)
    sub.add_argument("--leak", type=float, default=0.85)
    sub.add_argument("--forever", action="store_true")
    sub.add_argument("--tick-seconds", type=float, default=0.0)
    sub.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()
    if args.command == "download":
        download_data(args.data_dir, variant=args.variant, annotations=args.annotations, retinotopy=args.retinotopy)
    elif args.command == "prepare":
        print(json.dumps(prepare(args.data_dir, variant=args.variant,
                                 cache_dir=args.cache_dir), indent=2))
    else:
        run(args.cache_dir, args.checkpoint, steps=args.steps,
            stimulated_count=args.stimulated_count, gain=args.gain,
            leak=args.leak, forever=args.forever, tick_seconds=args.tick_seconds,
            checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
