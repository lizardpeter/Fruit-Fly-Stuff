# Fruit-Fly-Stuff 🪰

An open-source project exploring **persistent, experience-dependent learning** in mathematically simulated neural networks, with the longer-term goal of using real fruit fly connectomes.

> **Prototype status:** This version implements a **synthetic three-neuron circuit**, not the full fruit fly brain. It validates spiking dynamics, reward-driven learning, and persistent memory before we attempt biological reconstruction.

## What's implemented

- Sparse directed connections and simplified discrete-time spiking neurons (NumPy + SciPy).
- Membrane leakage, thresholded firing, reset, and one-step spike transmission.
- Reward-modulated plasticity with synaptic eligibility traces on selected edges.
- Atomic checkpoints including learned weights, membrane voltages, last spikes, eligibility traces, configuration, and simulation step count.
- An association experiment with separate save/restore and recall.
- Automated unit tests and a short GitHub Actions workflow.

## Run it

Python 3.11 or newer is required.

    python -m venv .venv
    source .venv/bin/activate
    # On Windows PowerShell: .venv\Scripts\Activate.ps1
    python -m pip install -e '.[dev]'
    python -m pytest -q
    python -m fruit_fly_stuff.demo --checkpoint state/toy-brain.npz

The default demo runs 6 paired stimulus/reward trials, saves a checkpoint, restores it, and shows that a conditioned cue now triggers an output. Expected relevant results:

    "before_training_cue_evokes_output": false
    "after_restoring_cue_evokes_output": true
    "learned_cue_weight": 1.18

## What this experiment demonstrates

The toy circuit has a conditioned cue neuron (0), an unconditioned input neuron (1), and an output neuron (2). The cue-to-output weight initially is 0.10 (too weak for a response). The unconditioned-input-to-output connection is 1.10. Repeated pairing with externally supplied positive reward raises the first weight. We then save **the network's numerical state**, load it into a new simulation object, and test whether the cue alone elicits an output spike. There is **no language model, gameplay scripting, or external memory database** behind the result.

A spike at time t affects downstream input at time t+1. Eligibility on a plastic edge accumulates when a presynaptic spike precedes a postsynaptic spike; reward alters eligible weights. The discrete steps and parameters have **not** been calibrated against fly physiology.

## Layout

- src/fruit_fly_stuff/simulation.py — sparse connection model, neural activity, plasticity
- src/fruit_fly_stuff/checkpoint.py — portable, non-pickle, versioned checkpoints
- src/fruit_fly_stuff/demo.py — learning and post-restart recall demo
- tests/ — checks for dynamics, conditioning, inhibitory edges, input validation, and persistence
- .github/workflows/ci.yml — bounded continuous-integration tests

## Research roadmap

1. Validate the small-circuit implementation, conditioning controls, and checkpoint compatibility.
2. Import a well-characterized real fly subcircuit, keeping the original neuron identifiers, edge data provenance, and neurotransmitter/sign assumptions.
3. Model relevant mushroom-body learning circuitry and compare responses with published biological observations.
4. Measure sparse propagation and plasticity costs on larger biological datasets; optimize before attempting a whole-brain run.
5. Connect anatomically justified sensory neurons and descending motor neurons to a simple virtual environment.
6. Compare behavior to randomized-connectome, plasticity-disabled, ablated, and trivial-controller baselines.

## Data sources to explore

- [FlyWire Codex](https://codex.flywire.ai/)
- [Janelia MaleCNS](https://male-cns.janelia.org/)
- [NEURON](https://neuron.yale.edu/neuron/) (biophysical simulator, not used here)

Biological datasets are not included and must be sourced according to their licenses. Synapse-contact counts are not the same as directed neuron-pair counts.

## Compute and persistence

CI is for **testing software**. GitHub Actions should not be used as an unrestricted general-purpose research computing cluster. Run long experiments elsewhere. Checkpoint files, huge data files, and pretrained data are intentionally Git-ignored; use durable external storage with appropriate provenance and backups.

## Scientific limitations

Real connectome wiring does not uniquely determine physiological behavior. Synthetic circuit learning is not evidence of biological fly intelligence or consciousness. Claims about the actual fruit fly brain require empirical validation and proper controls.

## License

Code: MIT (see LICENSE). Upstream research datasets retain their own licenses.
