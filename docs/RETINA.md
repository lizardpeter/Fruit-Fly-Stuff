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

The eye image is GPU rendered; **the optical sampling, temporal filtering
and entire neural network currently run on CPU**. Vulkan compute acceleration
can be implemented separately without altering the mapping format.

Sources: https://codex.flywire.ai/api/download ;
https://github.com/flyconnectome/flywire_annotations ;
https://flygym.readthedocs.io/latest/tutorials/vision_basics.html ;
https://pmc.ncbi.nlm.nih.gov/articles/PMC7588446/ ;
https://www.nature.com/articles/s41586-025-09276-5 .
