# Blender Extensions listing: Neuro3D Lab

Copy-ready text for the extensions.blender.org draft page. Every field matches the uploaded package (`blender_manifest.toml`, version 0.1.0).

| Field | Value |
|---|---|
| Name | Neuro3D Lab |
| Tagline (61 of 64 characters) | Virtual optical bench for diffractive optical neural networks |
| Type | Add-on |
| Version | 0.1.0 |
| Minimum Blender | 4.2.0 (tested on 4.5.14 LTS) |
| Licence | GPL-3.0-or-later |
| Maintainer | Francisco Angulo de Lafuente |
| Website | https://github.com/Agnuxo1/Neuro3D-Lab |
| Source code | https://github.com/Agnuxo1/Neuro3D-Lab |
| Issue tracker | https://github.com/Agnuxo1/Neuro3D-Lab/issues |
| Documentation | https://github.com/Agnuxo1/Neuro3D-Lab#readme |
| Tags | 3D View, Import-Export |
| Icon | `docs/extension/icon-256.png` (256 × 256 PNG) |
| Banner | `docs/extension/banner-1280x640.png` |
| Screenshot 1 | `docs/extension/screenshot-1-bench-1280x720.png` |

## Short description (one paragraph, for the listing summary)

Neuro3D Lab turns a Blender scene into a virtual optical bench. Place a source, phase layers and a detector, paint the phase heights as images, and export the bench to a documented JSON format. The numerical core (optlab, Apache-2.0) computes the coherent wave field, and the detector intensity is loaded back into Blender as an emissive image.

## Long description

**What it does**

- Builds a demo bench with a source plane, phase layers and a detector, with custom properties that define the optics.
- Exports the bench to `optlab.bench/1` (JSON plus height maps as `.npy`).
- Runs the simulation with the optlab numerical core and imports the detector intensity as an image.
- Traces rays through the scene to check that planes sit where the bench file says.

**What it does not do**

- It is a scalar, wave-based simulation. It does not model polarisation, reflections between planes or spectral dispersion.
- Blender is used for geometry and display. The physics is computed by optlab; the 3D view is only for visualisation.
- Propagation is valid only while each distance satisfies d < N·dx²/λ. The add-on refuses out-of-domain geometry unless the user opts in.

**Status and evidence**

- The numerical core is validated against analytic solutions (Gaussian beam with Gouy phase, Talbot self-imaging, on-axis aperture diffraction) and against a Rayleigh–Sommerfeld reference in the central region of the field. Some comparisons do not pass their pre-registered thresholds; the repository reports them in full.
- Digit classification with a phase-only network does not outperform digital baselines on the fresh test set reported in the repository.

Source, pre-registered contracts and all results: https://github.com/Agnuxo1/Neuro3D-Lab

## Changelog entry for 0.1.0

- First public release: demo bench, export and import of `optlab.bench/1`, in-domain validity check, self-test, documentation.

## Permissions (declared in the manifest)

- Files: save bench files to a user-chosen path.

## Notes for the reviewer

- The core library is separately licensed under Apache-2.0 (`src/optlab`), with the licence file in the repository root.
- The add-on needs the optlab package on the Python path. The self-test in `blender/selftest/` runs headless and checks the full round trip.
