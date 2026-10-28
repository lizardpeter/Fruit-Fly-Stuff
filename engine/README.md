# Fruit-Fly Engine — Rust + Vulkan

A minimal original 3D engine, not a wrapper around an existing game engine.

## Neural bridge (Milestone 2)

The Vulkan engine optionally receives neural motor commands from a local Python process and sends landmark-bearing proxies to it. See [Neural Bridge](../docs/NEURAL_BRIDGE.md) and run `Run-Neural-Fly.cmd` (toy) or `Run-Neural-Fly.cmd full`. **This now captures real offscreen Vulkan camera pixels and can use source-annotated optic and descending neurons; image-to-neuron mappings and motor effects remain experimental.** Scripted and keyboard modes remain when the bridge is disabled.

## Current implementation (Milestone 1)

- **Rust** game loop/world/scene update; low-level **Vulkan 1.3** renderer using `ash`.
- Winit creates a native Windows/Linux window; `ash-window` only creates the Vulkan surface.
- Real Vulkan graphics pipeline with WGSL compiled to SPIR-V at build time using `naga` (no installed Vulkan SDK or shader compiler required for building).
- GPU-drawn 3D cube-mesh terrain, markers, stylized fly body, wings, legs and compound-eye placeholders; Vulkan depth image, FPS-targeted redraw, and command-buffer/swapchain presentation.
- Human-controlled orbiting camera with perspective projection; keyboard motor control.
- `vision::EyeCamera`: independent **fly-facing camera specification** (position, view-projection, resolution, wide horizontal field of view) and `SensoryEncoder` trait for the eventual visual-to-neural interface. **A separate offscreen eye render pass and GPU-to-CPU readback now exist; biological calibration is still outstanding.**
- Headless deterministic world and sensor tests; Windows release build in GitHub Actions.

This first scene is a deliberately **stylized geometric placeholder**, not an anatomical/physical fly and not FlyGym. The autopilot is a scripted example, **not** intelligence emerging from FlyWire data. There are **no modeled flight, collisions, walking mechanics or vision neurons** yet.

## Run on Windows

**Easiest:** From [GitHub Actions – Rust Vulkan Engine](https://github.com/lizardpeter/Fruit-Fly-Stuff/actions/workflows/rust-engine.yml), open a successful run and download the `fruit-fly-vulkan-windows` artifact. Extract `fruit-fly-engine.exe` and double-click it.

**From source:** install Rust with [rustup](https://rustup.rs/) and a Vulkan-capable GPU driver, then from the repository directory:

```powershell
cargo run --release --manifest-path engine/Cargo.toml
```

Or double-click `Run-Vulkan-World.cmd` after installing Rust.

Runtime requires a **Vulkan 1.3-capable GPU and graphics driver**. The Vulkan SDK is optional for debugging, not needed to run this release.

### Controls

| Key | Action |
| --- | --- |
| P | Toggle scripted autopilot vs manual controls |
| W / S | Fly forward / backward (manual mode) |
| A / D | Turn left / right (manual mode) |
| Left / Right arrows | Orbit the viewer camera |
| Space | Pause/unpause world time |
| Esc | Quit |

Resize is intentionally disabled in the initial Vulkan swapchain implementation, because swapchain recreation is not implemented yet.

## Architecture and next milestones

- `src/world.rs`: time-stepped 3D positions, fly pose, declarative scene primitives, motor commands.
- `src/vision.rs`: fly-head camera calibration + typed frame encoding contract; does **not yet capture** visual data.
- `src/main.rs`: Vulkan instance/device/queue, swapchain, depth buffer, graphics pipeline, uniform-like push constants, frame synchronization, event loop.
- `shaders/scene.wgsl`: GPU geometry and lighting-tint primitives, compiled to SPIR-V by `build.rs`.

Planned in order:

1. Validate actual renderer launch on a PC with a Vulkan GPU; add GPU validation layers and fix any driver-specific issues.
2. Robust swapchain recreation, proper GPU buffer meshes, materials, camera orbit/zoom, glTF model support **without adopting an engine**.
3. Independent **offscreen fly-eye render pass** at a defined small resolution, asynchronous GPU-to-CPU readback; avoid accidentally giving the fly the external observer's camera.
4. Define exact sensory-neuron mapping for biological connectome indices; measure latency and encode image into stimulation (initially artificial with explicit provenance).
5. Map documented descending-neuron output into motor commands, and compare against scripted or random-controller controls.
6. Simulated environment physics, richer fly morphology, and checkpointing of **both** world and neural states.
7. Optional GPU compute for sparse neural graph updates via a separate Vulkan compute pipeline.

## Scientific and engineering caveats

The Python whole-brain simulator and Rust 3D world **do not yet communicate**. The fly is visibly shaped like a fly, but its behavior is currently scripted or keyboard-controlled. Real fly eyes are compound eyes, so a simple perspective camera is only a first sensor approximation. Do not call it photorealistic vision or autonomous biological behavior.

The CI verifies compiling and tests on Windows/Linux, but GitHub-hosted runners generally cannot show a physical Vulkan window. A local GPU is required for end-to-end interactive rendering verification.

## Viewing the fly-eye camera

When launched with `Run-Neural-Fly.cmd`, real GPU fly-camera pixels are also saved periodically to `results/fly-eye.bmp` (BMP, 128×96 pixels). This comes from an independent offscreen Vulkan render pass rather than the human spectator camera. The filename can be overridden with the `FRUIT_FLY_EYE_SNAPSHOT` environment variable. The integration workflow tests the output and attaches one BMP image as a downloadable GitHub Actions artifact.

### Vulkan compute optical sampler

The renderer now compiles a real WGSL compute shader to SPIR-V and dispatches it over the fly-eye offscreen texture each frame. Each of 1,442 facet invocations averages a nine-pixel optical neighborhood on the GPU; a small readback supplies the CPU's receptor kinetics. The full-brain spiking network remains on Python/CPU. GPU geometry plus photoreceptor sampling is **not** a calibrated retina, and no speedup is claimed without measurements.
