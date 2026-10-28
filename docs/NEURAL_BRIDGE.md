# Experiment D: linking the Vulkan fly to its neural simulator

## Run on Windows

1. Download/extract the repository. Install Python 3.11+.
2. Download a Windows `fruit-fly-engine.exe` from [Rust Actions](https://github.com/lizardpeter/Fruit-Fly-Stuff/actions/workflows/rust-engine.yml) and place it at the repository root. Or install Rust via rustup.
3. Double-click **Run-Neural-Fly.cmd**. This starts a fast **four-neuron synthetic** spiking-network controller.
4. For real FlyWire connections, open Command Prompt inside the repository and run **`Run-Neural-Fly.cmd full`**. The first invocation downloads and prepares FAFB v783 (may require 16–32 GB RAM).

The Rust renderer and Python neural worker are two **local** processes. They communicate over standard input/output using versioned JSON lines. They do not open network ports or require a cloud service.

## What is wired

3D fly position + heading → two visual feature signals from actual head-mounted Vulkan camera pixels → Python spiking-network input neurons → measured output spikes → Rust motor translator → fly position and heading.

- **Toy mode:** 4 synthetic neurons with two actual directed connections, serving as an easy integration test.
- **Full mode:** every neuron and all 15.1 million directed neuron pairs from the unfiltered FlyWire FAFB v783 sparse graph. To make the closed software loop testable, the default is **biologically annotated `optic` and `descending` neuron groups**, separated by source-provided left/right labels. We stimulate a limited subset of optic neurons and read descending spikes; both steps are uncalibrated. The previous strongest-edge probes remain available as a software baseline with `--mapping probes`. Selection is based on graph topology, not anatomical classification.
- The visual features come from an actual offscreen render pass into a fly-head camera, not hand-calculated landmark direction. The pixel feature extractor is artificial and does **not** implement compound-eye retinotopy.
- FlyWire classifications identify optic and descending neurons, but the actual pixel injection and spike-to-movement interpretation are experimental; the graph still has unvalidated all-positive normalized weights. This is not a model of biological sensory or motor pathways, fly-like autonomy, or consciousness.
- The spiking activity from the actual neural model controls movement in the Rust world; no motor spike values are fabricated by the bridge.

## Checkpoints, controls and timing

- Full mode stores/restores `state/bridge-full.npz`; toy mode uses `state/bridge-toy.npz`. Both are separate from the original standalone experiments. Checkpoints are written every 300 brain steps and upon clean exit.
- Rust sends up to 30 sensor requests per world second; the request channel has bounded capacity, and rendering does not block waiting for Python. Under load, sensory updates are dropped rather than queued indefinitely. This is **not** a guarantee of real-time neural simulation.
- Press **P** to toggle between the neural controller and manual **WASD**; Space pauses the world; the arrow keys orbit the viewer.
- The console logs the neural worker's readiness, network size, firing counts and errors. A failed worker does not silently switch back to scripted fly behavior.

## Implementation

- `src/fruit_fly_stuff/neural_bridge.py`: validates protocol and state, runs the real `WholeBrain` or 4-neuron synthetic circuit, checkpoints progress.
- `engine/src/bridge.rs`: spawns Python once, moves JSON messages on a worker thread, enforces version/sequence fields and avoids blocking the Vulkan loop.
- `engine/src/eye.rs`: renders and reads the fly's independent Vulkan eye; `engine/src/vision.rs`: extracts left/right pixel features.
- `engine/src/main.rs`: decodes returned neural spikes into forward/yaw commands and updates the world.

## Next biological milestone

Improve the current head-mounted camera into compound-eye optic sampling, use measured retinotopic mappings rather than arbitrary within-class neuron selections, and validate which descending neuron signals actually correspond to motor behavior. Compare real-connectome behavior with shuffled wiring and disconnected controls. GPU computation of the neural graph is still future work; the current full graph runs on the CPU.
