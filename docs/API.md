# API of `neuro3d_completion` (finite-support backend)

`src/neuro3d_completion` is the versioned backend added by the completion protocol (2026-10). It does not replace `optlab`; the legacy periodic model is unchanged, and `scripts/check_legacy_compatibility.py` checks that both agree where they overlap. Units are SI (metres) unless stated. Arrays are float64 / complex128.

## Conventions

- **Cell-energy amplitudes.** An input `u` of shape `(n, n)` holds one complex amplitude per physical pixel, with Σ|u|² equal to the energy. With refinement r, each pixel is split into r × r tied subpixels holding `u/r`, so the energy is unchanged. Detector scores sum cell energy over fixed physical rectangles: `s_k = β Σ M_k |f|²`.
- **Axes.** Row index increases downwards, `+y` is up, `+x` is right (as in `optlab`).
- **Three scales.** `n` physical pixels of pitch `pixel_m`; refinement `r` (quadrature, `pixel_m/r` sample pitch); padding `p` (FFT window `m = n·r·p`). Only `3·n²` phases are parameters, whatever `r` and `p` are.
- **Exterior.** `periodic` (legacy, p = 1 only), `transparent` (light outside the phase support passes), `aperture` (light outside the support is absorbed at every plane).
- **Transfer function.** `H = exp(2πi d √(λ⁻² − fx² − fy²))` for propagating components, 0 for evanescent ones; optional `phase_gradient` filter (an explicit conservative filter, not the Matsushima–Shimobaba algorithm).

## Core (`core.py`)

| Object | Purpose |
|---|---|
| `Optics(n, pixel_m, wavelength_m, distances_m, padding, refinement, boundary, filter_mode)` | Validated, frozen description; `m`, `sample_pitch_m`, `layers` |
| `detector_masks(n, pixel_m, regions=None)` | Ten disjoint half-open 4.8 mm squares by default (the historical layout) |
| `DiffractiveNetwork(optics, masks, beta, seed)` | NumPy model: `forward`, `loss_and_grad` (analytic adjoint gradient), `predict`, `fit` (Adam) |
| `score_error_bound(f, ε, masks, β)` | `β(2‖f‖ε + ε²)` for a valid L2 field-error bound ε |
| `certified_label(scores, error)` | Argmax certified only if the margin exceeds twice the score bound |

## Other modules

| Module | Main entry points |
|---|---|
| `torch_backend` | `TorchD2NN` (independent Torch FFT/autograd mirror), `digital_model(name)`, `check_equivalence` |
| `benchmark` | CLI `select` / `final` / `summarize`: DEV selection, frozen final runs with a TEST ledger, paired statistics |
| `data` | Verified MNIST / Fashion-MNIST admission, `encode` (28 → 64 linear zoom, unit energy) |
| `qmnist` | Writer-disjoint MNIST split from QMNIST metadata, QMNIST50K fresh test (`prepare`, `evaluate`, `summarize`) |
| `evidence` | Immutable records, Wilson, McNemar, Holm, crossed bootstrap, TEST reservation, contraction budget |
| `validation` | RS reference by quadrature, window (`boundary`) and refinement (`pixel`) convergence |
| `pixel_reference`, `rs_network` | Direct Rayleigh–Sommerfeld summation for one pixelated screen and for a whole network |
| `convergence` | Observed order and GCI (Celik et al. 2008) |
| `power` | Exact McNemar power, minimum detectable effect, paired TOST |
| `budget` | Measured seconds per training sample and protocol cost estimate |
| `expressivity` | Transfer matrix, quadratic forms, Jacobian rank, per-pixel influence |
| `photon_budget` | Photon-counting and NEP noise models; export of `region_energy` |
| `robustness` | Quantisation, phase noise, lateral shift, distance and wavelength perturbations |
| `hardware` | Analysis of calibrated measurements (W) against frozen simulations; no fitted gain |
| `interfaces` | Fresnel coefficients with evanescent transmission amplitude in TIR |
| `bench_adapter` | `simulate_bench_v2` for `optlab.bench/1` files (point-sample convention, see guide E12) |
| `claims`, `blind`, `guards` | Claim register and wording audit; blind analysis; refusal of reserved splits |
| `maxwell_probe` | Guide's Meep probe (requires `pymeep`, not installed with the package) |

## Project tools (`scripts/completion_tools/`)

`select_sharded.py` (DEV selection, one process per grid point, identical to `benchmark select`), `fidelity_study.py` (CONTRATO-014), `verify_frozen.py`, `q50k_verdicts.py`, `window_check.py`, `maxwell_validation.py` (CONTRATO-015), `queue_driver.py` and `make_jobs.py` (workstation orchestration), `gpu_probe.py`, `compare_replay.py`.

## Typing

The package has type-free NumPy signatures; it does not ship `py.typed`, because full static typing is not offered.
