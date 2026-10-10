# Blender Extensions submission form: exact text and files

Form order, as shown on the draft page. Every value below is final for version 0.1.1 (bundled optlab).

## 1. Description (Markdown)

Neuro3D Lab turns a Blender scene into a virtual optical bench. Place a source plane, phase layers and a detector, paint the phase heights as images, and export the bench to a documented JSON format (`optlab.bench/1`). The numerical core computes the coherent scalar wave field, and the detector intensity is loaded back into Blender as an emissive image.

**What it does**
- Builds a demo bench with a source, phase layers and a detector, defined by custom properties.
- Exports the bench (JSON plus height maps as `.npy`), runs the simulation and imports the detector intensity.
- Refuses geometry outside the validity limit of plain angular-spectrum propagation, unless you opt in.

**What it does not do**
- It is a scalar wave model. It does not model polarisation, reflections between planes or spectral dispersion.
- Blender provides geometry and display. The physics is computed by the bundled optlab core.

**Status**
- The core is checked against analytic solutions (Gaussian beam with Gouy phase, Talbot self-imaging, on-axis aperture diffraction). Some comparisons do not pass their pre-registered thresholds, and the repository reports them in full.
- On a fresh, writer-disjoint digit test set, the phase-only network does not outperform a small digital baseline. Details are in the repository.

optlab (Apache-2.0) is bundled inside the add-on, so the simulation works after installation from the catalogue.

Source, pre-registered contracts and all results: https://github.com/Agnuxo1/Neuro3D-Lab

## 2. URL for reporting issues or support

https://github.com/Agnuxo1/Neuro3D-Lab/issues

## 3. Initial Version: release notes (Markdown)

**0.1.1**, first public release.

- Demo optical bench: source plane, phase layers and detector, defined by custom properties.
- Export to `optlab.bench/1` (JSON plus height maps as `.npy`) and import of the simulated detector intensity as an image.
- Validity check: geometry outside the propagation limit is refused unless explicitly allowed.
- Bundled optlab core (Apache-2.0), so no separate installation is needed.
- Headless self-test (15 of 15 steps pass with Blender 4.5.14).

## 4. Featured image (16:9, at least 1920 × 1080)

`docs/extension/featured-1920x1080.png` (1920 × 1080, PNG, 1.1 MB)

## 5. Icon (256 × 256 PNG)

`docs/extension/icon-256.png`

## 6. Previews (16:9; add all three)

1. `docs/extension/preview-1-bench-1920x1080.png`. Caption: "Demo optical bench rendered in Blender with the simulated detector intensity."
2. `docs/extension/preview-2-validation-1920x1080.png`. Caption: "Gaussian beam: numerical propagation against the exact solution (maximum relative width error 3.5 × 10⁻⁵)."
3. `docs/extension/preview-3-pipeline-1920x1080.png`. Caption: "Workflow: Blender scene, optlab.bench/1 export, optlab numerical core, intensity back in Blender."

## 7. Description image (optional)

`docs/extension/preview-3-pipeline-1920x1080.png`

## Package to upload

`D:/PROJECTS/.cognition/neuro3d-lab/dist_bundle/neuro3d_lab-0.1.1.zip` (version 0.1.1, bundles optlab, GPL-3.0-or-later). Replace the 0.1.0 draft package, which does not include optlab.
