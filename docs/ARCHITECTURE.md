# Architecture

Neuro3D-Lab separates the **geometry** (Blender) from the **physics** (Python/numpy). Blender is the optical bench: planes are placed there, phase heights are painted there, and results are viewed there. The field is computed by `optlab`. Everything in this repository is simulation; no physical hardware experiment exists.

```
Blender (bench)                         optlab (physics)                 Blender (results)
───────────────                         ────────────────                 ─────────────────
SOURCE, LAYER_k, DETECTOR  ──export──▶  bench.json + *_height.npy  ──▶  simulate_bench()
heights (float image)                                                   │
                                                                        ▼
intensity image  ◀──import───────────  detector_intensity.npy  ◀────────┘
```

## Modules

| Path | Responsibility |
|---|---|
| `src/optlab/units.py` | Wavenumber, phase from optical path length (OPL), optical path |
| `src/optlab/propagation.py` | Angular spectrum: `transfer_function`, `angular_spectrum`, adjoint |
| `src/optlab/elements.py` | Phase elements with absorption, detector masks |
| `src/optlab/fresnel.py` | Normal and oblique Fresnel coefficients, total internal reflection (TIR) |
| `src/optlab/interferometer.py` | Beam splitter, phase shifter, unitary Mach-Zehnder interferometer (MZI) |
| `src/optlab/d2nn.py` | Diffractive network: propagation, loss, analytic gradient, Adam |
| `src/optlab/bench.py` | Reading and validation of `optlab.bench/1`, bench simulation |
| `src/optlab/stats.py` | Wilson interval |
| `src/optlab/registry.py` | Run registry in JSONL, append-only |
| `blender/neuro3d_lab/` | Add-on: build the demo scene, export, import, ray tracing |
| `blender/selftest/run_selftest.py` | Headless test of the add-on and of its contract with optlab |
| `tests/` | Tests for each module, with thresholds from `docs/CONTRACTS/` |

## Format `optlab.bench/1`

```json
{
  "schema": "optlab.bench/1",
  "wavelength_m": 6.33e-7,
  "medium_index": 1.0,
  "grid": {"n": 64, "pixel_m": 8e-6},
  "source": {"z_m": 0.0},
  "layers": [
    {"name": "LAYER_1", "z_m": 0.02, "n_medium": 1.0, "n_material": 1.5,
     "kappa": 0.0, "height_npy": "LAYER_1_height.npy", "height_max_m": 1e-6}
  ],
  "detector": {"z_m": 0.06,
               "regions": [{"label": 0, "cx_m": 0.0, "cy_m": 0.0, "w_m": 1e-4, "h_m": 1e-4}]},
  "output": {"intensity_npy": "detector_intensity.npy"}
}
```

Rules:

- The `z_m` values must strictly increase: source < layers < detector.
- `medium_index` (optional, default 1.0) is the refractive index of the medium between the planes. Propagation between planes uses the wavelength `wavelength_m / medium_index`. The phase of each layer uses the vacuum wavelength `wavelength_m` and the difference `n_material − n_medium`. Every layer's `n_medium` must equal `medium_index`; otherwise a `ValueError` is raised.
- The distance between two planes is the difference of their `z_m` values.
- The real height is `height_npy · height_max_m`, with shape `(n, n)`.
- Relative paths are resolved from the directory that contains the JSON file.
- The source may give `amplitude_npy` (shape `(n, n)`). Without it, the initial field is a plane wave of amplitude 1.

## Domain of validity

Plain angular-spectrum propagation is reliable only while every free-space segment is shorter than

```
z_max = N · pixel_m² / λ_medium        with λ_medium = wavelength_m / medium_index
```

where `N` is the grid size. Equivalently, `d < N · dx² / λ`, with `λ` the wavelength in the medium. The free-space segments are: source to the first layer, each layer to the next one, and the last layer to the detector. Beyond `z_max`, aliasing invalidates the result (see `docs/MODEL.md`, section 2).

`simulate_bench` checks every segment before it propagates the field:

- A segment with `d >= z_max` raises `ValueError`. The message names the segment, its length and `z_max`.
- With `allow_out_of_domain=True`, each such segment emits a `warnings.warn` with the same information, and the simulation continues. A result obtained this way lies outside the validity domain and must be reported as such.

The Blender demo bench (N = 256, pixel 8 µm, λ = 633 nm, source at 0 mm, layers at 20 mm and 40 mm, detector at 60 mm) lies inside this domain. The self-test checks the demo against the limit.

## Image orientation convention (Blender ↔ numpy)

Blender stores images with the origin at the bottom left. The exporter in `blender/neuro3d_lab/bench_io.py` converts them to numpy arrays with this rule, and the importer applies the inverse:

- `a[i, j]`: `i` is the row counted from the **top** of the image; `j` is the column counted from the left. The array is `np.flipud` of Blender's pixels.
- Physical coordinates on a plane: `+x` = increasing column (to the right); `+y` = up, that is, decreasing row. Row 0, the top of the image, has the largest `y`.
- `optlab.elements.detector_masks` uses the same rule: `x_j = (j − (n−1)/2)·pixel`, `y_i = −(i − (n−1)/2)·pixel`.
- The exported height is in `[0, 1]`. The real height is `height_npy · height_max_m`.

This convention was fixed after an erratum: the initial 0.1 version had `+y` pointing down in `detector_masks`. `tests/test_orientation.py` guards against a regression.

## Blender add-on

Layout:

```
blender/
├── neuro3d_lab/                 add-on (GPL-3.0-or-later)
│   ├── __init__.py              bl_info, register() and unregister()
│   ├── blender_manifest.toml    extension manifest (Blender 4.2 or later)
│   ├── LICENSE                  GPL-3.0 license text for this folder
│   ├── bench_io.py              scene building, export, intensity import, ray tracing
│   └── operators.py             operators, scene properties and the N-panel
├── selftest/run_selftest.py     headless self-test (run by hand)
└── render_bench_gif.py          renders the bench GIF
```

- **License.** The core library `optlab` is Apache-2.0 (root `LICENSE`). The add-on is GPL-3.0-or-later (its own `LICENSE`, and the `license` field of the manifest). The reason is that the official Blender extensions catalogue requires GPL-3.0-or-later for add-ons, and the add-on imports `optlab`.
- **Manifest.** `blender_manifest.toml` holds the add-on id, version, `blender_version_min`, the SPDX license identifier, and the justification shown to users for each permission (saving bench files to a user-chosen path). Validate it with `blender --command extension validate blender/neuro3d_lab`.
- **Units.** JSON values are in metres. Blender units are converted with the scene's `unit_settings.scale_length`.
- **Visualisation.** `blender/render_bench_gif.py` magnifies each plane 15 times in x and y, for visibility only. The z spacing is physical.
- **Self-test.** `blender/selftest/run_selftest.py` runs headless with a local Blender 4.5 binary. It is run by hand, not in CI. Its output goes to `<out>/selftest`, where `<out>` is the output folder described below.

## Output folder

`NEURO3D_LAB_OUT` sets the base output folder. If it is not set, the base is `<system temp>/neuro3d-lab`. Subfolders:

- Blender add-on default output: `<base>/bench`.
- Self-test: `<base>/selftest`.
- Bench GIF renderer: `<base>/render/frames`.
- Experiments: `<base>/results`.

Large or raw outputs stay outside the repository. Only small summaries are committed, for example `experiments/results/validation_summary.json`.

## Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request to `master`. Its `pytest` job runs on Ubuntu with Python 3.13, installs the package with `pip install -e ".[dev]"`, and runs `pytest -q`.

**Blender is not run in CI.** The add-on self-test and the manifest validation are run locally with a Blender 4.5 binary (see `CONTRIBUTING.md`).

## Why not Cycles or Eevee

Cycles and Eevee trace rays or rasterise, but they do not keep coherent phase along paths. An interferometer or a diffractive network must add complex fields with their phase. That is why Blender provides the geometry and the visualisation, and `optlab` computes the field.
