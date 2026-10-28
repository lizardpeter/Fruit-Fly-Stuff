# Experiment D: linking the Vulkan fly to its neural simulator

## Run on Windows

1. Download/extract the repository. Install Python 3.11+.
2. Download a Windows `fruit-fly-engine.exe` from [Rust Actions](https://github.com/lizardpeter/Fruit-Fly-Stuff/actions/workflows/rust-engine.yml) and place it at the repository root. Or install Rust via rustup.
3. Double-click **Run-Neural-Fly.cmd**. This starts a fast **four-neuron synthetic** spiking-network controller.
4. For real FlyWire connections, open Command Prompt inside the repository and run **`Run-Neural-Fly.cmd full`**. The first invocation downloads and prepares FAFB v783 (may require 16–32 GB RAM).

The Rust renderer and Python neural worker are two **local** processes. They communicate over standard input/output using versioned JSON lines. They do not open network ports or require a cloud service.

## What is wired

3D fly position + heading → two geometric landmark-bearing signals → Python spiking-network input neurons → measured output spikes → Rust motor translator → fly position and heading.

- **Toy mode:** 4 synthetic neurons with two actual directed connections, serving as an easy integration test.
- **Full mode:** every neuron and all 15.1 million directed neuron pairs from the unfiltered FlyWire FAFB v783 sparse graph. To make the closed software loop testable, two strong **nonoverlapping directed edges** are selected as artificial left/right input/output probe pairs. Selection is based on graph topology, not anatomical classification.
- The visual signals come from a simple bearing-and-distance calculation for two colored world landmarks; **the Vulkan fly-eye camera does not yet capture visual pixels.**
- The full graph has unvalidated all-positive normalized weights. This is not a model of biological sensory or motor pathways, fly-like autonomy, or consciousness.
- The spiking activity from the actual neural model controls movement in the Rust world; no motor spike values are fabricated by the bridge.

## Checkpoints, controls and timing

- Full mode stores/restores `state/bridge-full.npz`; toy mode uses `state/bridge-toy.npz`. Both are separate from the original standalone experiments. Checkpoints are written every 300 brain steps and upon clean exit.
- Rust sends up to 30 sensor requests per world second; the request channel has bounded capacity, and rendering does not block waiting for Python. Under load, sensory updates are dropped rather than queued indefinitely. This is **not** a guarantee of real-time neural simulation.
- Press **P** to toggle between the neural controller and manual **WASD**; Space pauses the world; the arrow keys orbit the viewer.
- The console logs the neural worker's readiness, network size, firing counts and errors. A failed worker does not silently switch back to scripted fly behavior.

## Implementation

- `src/fruit_fly_stuff/neural_bridge.py`: validates protocol and state, runs the real `WholeBrain` or 4-neuron synthetic circuit, checkpoints progress.
- `engine/src/bridge.rs`: spawns Python once, moves JSON messages on a worker thread, enforces version/sequence fields and avoids blocking the Vulkan loop.
- `engine/src/world.rs`: generates two landmark-bearing **proxy** sensations.
- `engine/src/main.rs`: decodes returned neural spikes into forward/yaw commands and updates the world.

## Next biological milestone

Implement a separate head-mounted Vulkan render-to-texture eye, sample its pixels, map outputs to anatomically justified optic-lobe neuron IDs, and interpret validated descending motor pathways. Compare real-connectome behavior with shuffled wiring and disconnected controls. GPU computation of the neural graph is still future work; the current full graph runs on the CPU.
