# Fruit-Fly-Stuff 🪰

An open-source project exploring **persistent, experience-dependent learning** in mathematically simulated neural networks, with the longer-term goal of using real fruit fly connectomes.

> **Prototype status:** This version implements a **synthetic three-neuron circuit**, not the full fruit fly brain. It validates spiking dynamics, reward-driven learning, and persistent memory before we attempt biological reconstruction.

## Experiment D — neural network controls the Vulkan fly (prototype)

**[Link the 3D fly and neural network](docs/NEURAL_BRIDGE.md):** Rust and Python now exchange geometric landmark inputs and measured neuron spikes through a local JSON-lines bridge. Double-click `Run-Neural-Fly.cmd` (fast 4-neuron toy), or use `Run-Neural-Fly.cmd full` for the **entire FlyWire 139,255-neuron graph**. The full graph runs genuine sparse propagation but currently uses **artificial topological probes**, not identified biological sensory/motor circuits. The fly now has a **real offscreen Vulkan eye camera**, whose pixels produce visual features. Full mode can use **FAFB v783 optic and descending neuron annotations**, but pixel-to-neuron and neural-to-motor mappings are still experimental.

## Experiment C — custom Rust + Vulkan 3D fly world

A **separate original 3D engine** now lives in [`engine/`](engine/) and does **not** use Unity, Unreal, Bevy, wgpu, or FlyGym. It uses raw [Vulkan bindings via ash](https://github.com/ash-rs/ash), native [winit](https://github.com/rust-windowing/winit) window events, and custom Rust world/scene/renderer code.

**Initial milestone:** Vulkan 1.3 GPU render pipeline and depth buffer, a stylized articulated fly in a small 3D arena, a camera for *the human observer*, keyboard steering, a clearly labeled scripted autopilot, and a **separate fly-eye camera/sensory encoding interface**. This is still a placeholder model, not an anatomically accurate fly, neural control, or captured visual input.

- [Detailed engine setup and status](engine/README.md)
- [Windows Rust build/test Actions](https://github.com/lizardpeter/Fruit-Fly-Stuff/actions/workflows/rust-engine.yml)
- On Windows, run `Run-Vulkan-World.cmd` from the repository, or download the compiled Windows .exe from a successful Actions run.

**Important:** rendering is accelerated by Vulkan, but the existing whole-connectome **neural computation remains Python/CPU**. GPU neural simulation and image-to-neuron wiring are future, independent features.

## Earlier simulation experiments

**A. [Whole FlyWire connectome on your PC](docs/WHOLE_BRAIN.md):** downloads FAFB v783, keeps all 139,255 neuron IDs and all neuron-pair connections in the *unfiltered Codex contact-count table*, then runs a compact sparse mathematical model. It resumes its voltages and spikes from a local checkpoint. **This does not yet model realistic fly physiology, plasticity, behavior or consciousness.**

**B. [Three Hodgkin–Huxley biophysical neurons](docs/THREE_NEURONS.md):** a separate three-cell electrical simulation with sodium/potassium ion channels, synaptic excitation/inhibition, and CSV/PNG membrane-voltage traces.

All recorded connections means the released *per-pair counts*; this is not a per-synapse geometry reconstruction. See the separate experiment documentation and scientific caveats.

### Windows quick start (two independent programs)

- Double-click `Run-Whole-Brain.cmd` to download, prepare, and run the full sparse connectome continuously with checkpoints (**16+ GB RAM recommended for preprocessing**).
- Double-click `Run-Three-Neurons.cmd` to simulate three detailed membrane models and save voltage plots locally.

Both scripts create a virtual environment as needed. Requires Python 3.11+ with the `py` Windows launcher installed. You can run either experiment without the other.

## What's implemented

- Sparse directed connections and simplified discrete-time spiking neurons (NumPy + SciPy).
- Membrane leakage, thresholded firing, reset, and one-step spike transmission.
- Reward-modulated plasticity with synaptic eligibility traces on selected edges.
- Atomic checkpoints including learned weights, membrane voltages, last spikes, eligibility traces, configuration, and simulation step count.
- An association experiment with separate save/restore and recall.
- Automated unit tests and a short GitHub Actions workflow.

## Run in GitHub Actions — no local setup

1. Open **[Run Brain Experiment](https://github.com/lizardpeter/Fruit-Fly-Stuff/actions/workflows/brain-experiment.yml)** in GitHub Actions.
2. Select **Run workflow**, ensure **main** is selected, and choose **3 trials** (the default). Leave **reset** unchecked.
3. Start the workflow; when it finishes, open the run's **Summary** for learned weights and recall results.
4. Run it again to load the previous run's checkpoint and continue learning.

The first run starts with a cue weight of **0.10**; 3 learning trials increase it to **0.64** (not yet sufficient to activate the output neuron). The second run restores the saved state, trains for 3 more trials, and increases it to **1.18**, after which the cue alone activates the output neuron. This is a reproducible toy example of experience-dependent learning across different ephemeral runner machines.

### Persistent state and artifacts

On its first run, the workflow creates a separate **simulation-state** branch with the small `state/toy-brain.npz` checkpoint. Later runs restore that state and commit its updated weights and neural state back to the same branch. Runs are serialized so two runs cannot race to overwrite the checkpoint. Each run also uploads the checkpoint and `results/experiment.json` as a 30-day GitHub Actions artifact, and includes a human-readable table in the Actions run summary.

Selecting **reset** restarts the toy circuit and replaces the saved checkpoint. State history remains in Git until it is deleted. Do not use this small-file Git approach for full-brain models or large datasets; use dedicated object storage for those. These bounded workflows are for developing and validating the simulator, not indefinite scientific compute on hosted Actions.

**Scientific caveat:** this demo models three synthetic neurons, not the fly's actual connectome. The biologically informed simulation is future work.

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
