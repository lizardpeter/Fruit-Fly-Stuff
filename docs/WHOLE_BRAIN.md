# Experiment A — all recorded FAFB v783 neurons and neuron-pair connections

This uses the **real FlyWire FAFB v783 connectivity graph** (139,255 neurons).
It is a **simplified mathematical point-neuron simulation**, *not* a digital
reconstruction of a living fly. There is no virtual body, sensory physiology,
synaptic learning, calibrated behavior, or consciousness claim*.

## Windows PowerShell (Python 3.11+)

In the repository directory:

    py -3 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -e .
    .\.venv\Scripts\python.exe -m fruit_fly_stuff.whole_brain download
    .\.venv\Scripts\python.exe -m fruit_fly_stuff.whole_brain prepare
    .\.venv\Scripts\python.exe -m fruit_fly_stuff.whole_brain run --steps 500

To let it run until Ctrl+C (and automatically checkpoint its state):

    .\.venv\Scripts\python.exe -m fruit_fly_stuff.whole_brain run --forever --checkpoint-every 500

Each subsequent run with the same parameters and graph **restores**
`state/whole-brain.npz`. To restart from scratch, rename or move the
checkpoint (do not delete valuable learned work). This initial whole-graph
model preserves voltages and spike flags but **does not yet learn**: its weights
are fixed. Persistent state != synaptic learning.

### No automatic download?

The public objects are at
`https://storage.googleapis.com/flywire-data/codex/data/fafb/783/`.
If the provider changes access or names, sign in to
[Codex Download Data](https://codex.flywire.ai/api/download), choose **FAFB v783**,
and place these two files (unchanged) in `data/fafb783/`:

- `neurons.csv.gz` — all neuron root IDs
- `connections_no_threshold.csv.gz` — **unfiltered** neuron-pair connections

Then execute **prepare** and **run** above. Do not mix FAFB versions or substitute
a Princeton source table without verifying naming and provenance.

A smaller optional dataset is available: pass `--variant filtered` to
both download and prepare. It drops weak pair connections and therefore is
**not** the all-connections option.

### What exactly is imported?

- Every root ID, even disconnected neurons; IDs are preserved as uint64.
- Every positive synaptic contact count listed in the chosen export.
- Rows for a neuron pair in different brain regions are summed into one
  directed neuron-pair contact count.
- The matrix is CSR sparse, so we do **not** allocate 139,255 × 139,255 cells.
- Per-postsynaptic-neuron contact counts are normalized to sum to one.
  **This is an illustrative assumption**; synapse counts alone are not
  biological strengths. The current engine assumes every pair is excitatory,
  even though real connections include inhibition and modulation.
- `processed/metadata.json` records dataset, counts, filter setting,
  and SHA-256 hashes of the source files. `processed/root_ids.npy`
  maps internal indices back to actual neurons.

**Unfiltered** means all connections in the released *per-neuron contact
summary*. It does not load microscopic individual synapse coordinates or
neuronal morphology (those are different, much larger datasets). The release
itself has finite proofreading and segmentation coverage; "all" refers to
the neurons and connections in this specific release.

### Resource expectations and safety

Prepare downloads ~214 MB compressed (source-dependent), then processes
millions of rows. Processing may temporarily use **multiple GB of RAM**,
especially the unfiltered dataset. Start with **16 GB RAM or more and
several GB free disk**, ideally 32 GB for headroom. No GPU is needed.
Actual time and memory use must be benchmarked; no real-time guarantee.

The program prints the active neuron count every 25 simulation steps and
saves a checkpoint automatically every 500 steps and on Ctrl+C. A GitHub
Actions runner is **not** used for the full data import or indefinite running.

### Biological next steps

Neurotransmitter-specific synaptic signs, neuron-specific parameters,
anatomically grounded sensory stimulation, plasticity, motor decoding,
a virtual body, control baselines, and experimental comparisons.
