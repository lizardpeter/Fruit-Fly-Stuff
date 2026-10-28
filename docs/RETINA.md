# Retina-to-Connectome Compiler — experiment E

This first working retina model uses the existing fly-mounted Vulkan camera,
a Rust hexagonal facet sampler and a Python FAFB v783 column compiler.

**Scope:** FlyWire root IDs, cell types, hemispheres, column coordinates and
synaptic connections are from the real v783 release. The conversion between
virtual facet coordinates and real visual columns is **not scientifically
calibrated** and must not be presented as a faithful fly retina.

## Pipeline

1. Vulkan renders the world through the fly's head-mounted 128×96 camera.
2. Rust groups the image into **721 projected hex facets per eye half**,
   sampling each with a 9-pixel optical acceptance kernel.
3. Eight receptor-like graded channels per facet: R1–R6 broad-band RGB
   proxies; R8 blue/green pale/yellow proxy; R7 **UV unavailable** because
   ordinary RGB cannot measure UV. Deterministic 30% pale assignment is
   illustrative, not inferred for real ommatidia.
4. Fast response (25 ms) and slower light adaptation (200 ms) produce
   temporally dependent graded signals. These values are not biological
   parameter estimates.
5. All 1,442 facet response values travel across the threaded Rust/Python
   subprocess bridge, alongside the legacy left/right aggregate.
6. The compiler reads real FAFB v783 column assignment records with fields
   root_id, hemisphere, type, column_id, x, y, p, q; it selects the L1/L2
   cell classes and maps the columns to virtual facets.
7. The full Python connectome receives stimulation of individual mapped
   **real neuron indices**, with the prior experimental descending-neuron
   decoder providing motor signals.

## Windows quick start (approximate alignment)

Download/extract the latest repository and the latest Windows Vulkan engine
executable from GitHub Actions into its root. Install Python 3.11+; then
double-click **Run-Retina-Fly-Approximate.cmd**.

The script downloads the v783 graph, classification and column tables,
prepares the sparse network and **explicitly** invokes --approximate to
build data/fafb783/processed/retinotopy.json. It launches full-network mode
with FRUIT_FLY_NEURAL_MAPPING=retinotopic. The file stores source SHA-256,
processed graph SHA-256, real root IDs and experimental facet assignments.

## Supply a calibration instead

Use a CSV of known column-to-facet correspondences with these headers:

    hemisphere,column_id,facet_index

For instance: left,97,123 and right,355,890. Then run:

    python -m fruit_fly_stuff.retinotopy --calibration your_calibration.csv

A user supplied calibration is validated for format and provenance but is not
independently experimentally verified. Cross-eye mappings and stale graph
versions are rejected. Without --approximate or --calibration, the compiler
refuses to guess.

## Limitations

The current 721-per-half layout is NOT a true 270-degree dual compound-eye
rendering, nor does the FlyWire column table alone determine optical
orientation or retinotopic chirality. The virtual->L1/L2 correspondence is
an exploratory hypothesis. Fly photoreceptors frequently use graded
signals, but the current whole graph uses a simplified spike-threshold
model. The motor decoder remains an uncalibrated placeholder.

The eye image is GPU rendered; **Vulkan compute samples all 1,442 projected facet locations** on the GPU. Rust performs stateful receptor/adaptation updates on CPU after reading back only the facet RGB values. The Python whole-connectome neural propagation remains CPU. These are not yet benchmarked against GPU compute alternatives.

Sources: https://codex.flywire.ai/api/download ;
https://github.com/flyconnectome/flywire_annotations ;
https://flygym.readthedocs.io/latest/tutorials/vision_basics.html ;
https://pmc.ncbi.nlm.nih.gov/articles/PMC7588446/ ;
https://www.nature.com/articles/s41586-025-09276-5 .

### ON/OFF retinal transients (fix)

The previous sampler rectified all decreases in luminance to zero, losing dark edges and leaving L2 unable to receive a separate OFF signal. The updated bridge transmits two **1,442-element** arrays: positive transients (ON) and negative transients (OFF). The experimental mapping routes L1-class neurons from ON and L2-class neurons from OFF, reflecting the better-known upstream ON/OFF pathway distinction, while explicitly *not* claiming to simulate the true graded L1/L2 membrane potential or validated retinal wiring. Strongly static scenes still adapt: this is not a complete persistent luminance or motion model.

### Timebase correction

The photoreceptor response now advances using elapsed world-simulation time, not a hard-coded 60 Hz assumption. Pausing the world freezes receptor/adaptation state. This improves consistency across computers but still does not establish measured fly kinetics.

### Checkpoint compatibility and neural persistence

Full-brain checkpoints are now separated by mapping mode: `state/bridge-full-classified.npz`, `state/bridge-full-probes.npz`, and `state/bridge-full-retinotopic.npz`. The checkpoint also records a fingerprint of the graph, mapping mode, source classification data, and retina manifest. A different visual calibration fails restoration rather than silently reusing incompatible neural state. Earlier standalone whole-brain checkpoints remain compatible with the standalone simulator, but old bridge checkpoint filenames are not automatically migrated.

### Optical GPU/CPU correctness and dark scenes

The Vulkan workflow now enables `FRUIT_FLY_RETINA_VERIFY_GPU=1` for an actual 1,442-facet shader-versus-CPU numerical comparison on the first rendered frame. The CPU reference decodes sRGB when the offscreen render target has an sRGB format. Any discrepancy above 0.012 causes a failing runtime test. A fully black image is valid; the previous non-black requirement has been removed. Full-image GPU-to-host copies and CPU readback happen only for explicit eye snapshots or the parity test; ordinary sensory updates use only the small Vulkan compute-buffer readback. This test checks software consistency, **not biological accuracy**.

### Deterministic visual-event experiment

For a test of actual environmental input rather than an assumed neuron spike, set `FRUIT_FLY_VISION_TRIAL=1` and `FRUIT_FLY_SMOKE_FRAMES=240` in the Vulkan engine environment. The world briefly displays a white bar directly ahead of a stationary fly (time 0.75–1.50 s). The real GPU-rendered camera, Vulkan optical sampler and stateful photoreceptor model must generate both an ON transient when the object appears and an OFF transient when it disappears. GitHub Actions asserts both signals are present. This tests the conversion chain, **not** the biological correctness of its calibration or a fly behavior.

### Display frame pacing (world-time regression)

The engine now schedules redraws against the previous frame's time instead of calling `request_redraw` repeatedly in its idle callback. The old loop bypassed its 16 ms wait deadline and could render hundreds of frames in under a second, badly distorting the correspondence between CI frame counts and simulated visual events. The physical visual-event regression now also checks that simulation time has progressed far enough to show both transitions.

### Projected facet layout v2

The original axial-to-pixel formula accidentally used a denominator three times larger than the lattice extent, so each eye sampled an unnecessarily narrow band and missed much of the camera image, including parts of the forward center. The v2 normalization covers nearly the full source half-image (with a small edge margin). The facet IDs and FAFB root IDs are unchanged, but the **pixel-to-facet geometry is different**, so the compiled mapping layout ID is bumped to `two-721-projected-hex-halves-v2`; old v1 mapping manifests must be regenerated, and mismatched neural checkpoints will be rejected. This is still a projected approximation, not measured compound-eye calibration.

### v2 upgrade without overwriting old neural state

The new retinal layout saves to `state/bridge-full-retinotopic-v2.npz`. Earlier `state/bridge-full-retinotopic.npz` files remain untouched and cannot be silently restored with different sensory geometry. The new v2 file still has mapping fingerprints and will reject incompatible recalibrations. Other controller filenames are unchanged.
