# Changelog

All notable changes. Dates are UTC+2. Historical records are never rewritten; corrections are added as new entries or addenda.

## [0.2.1] — 2026-10-13
- Blender Extensions review fix: relative imports for both bundled libraries, no `sys.path` changes, no external source-path option, and writable user-data output defaults.
- Bundled completion adapter changes only its import of `load_bench`; source numerical code remains unchanged. The vendor identity test explicitly checks this one permitted adaptation.
- Added namespace/collision regression tests and a clean-profile installed-package check that asserts unchanged paths and untouched global module names.
- Uploaded to Blender Extensions with English description and release notes; approval pending. Evidence: `evidence/blender-review-20261013/` (100 pytest tests; Blender 4.5.14 LTS: 15/15 legacy, 8/8 completion, validation/build/install).

## [Unreleased] — completion protocol (2026-10-12 to 2026-10-13)

### Added
- `src/neuro3d_completion/`: finite-support backend (fixed physical phase support, tied-subpixel quadrature, FFT window, `periodic` / `transparent` / `aperture` exteriors), Torch mirror, benchmark protocol with TEST ledger, data admission by SHA-256, writer-disjoint MNIST and QMNIST50K, direct Rayleigh–Sommerfeld references, convergence (observed order, GCI), power analysis, expressivity, robustness, noise budgets, hardware analysis, claim register, blind analysis (completion guide v1 + v2, 42 files integrated without replacing any file).
- Contracts CONTRATO-012 (numerical closure record), CONTRATO-013 (closure benchmark), CONTRATO-014 (discretisation fidelity) and CONTRATO-015 (2D Maxwell check), the last three pushed before their runs with hash manifests.
- `scripts/completion_tools/`: sharded DEV selection (identical to `benchmark select`, tested), fidelity study, frozen-code check, writer-grouped verdicts, window check, Maxwell validation, refinement 4-8-16 on the GPU, queue driver and job lists, GPU probe, replay comparison.
- `evidence/`: every completion record (compatibility, replays, convergence, RS, GCI, power, GPU throughput, JEV decisions, Blender logs, protocol manifests).
- `examples/` (four runnable examples), `docs/API.md`, `docs/LITERATURE_REVIEW.md`, `docs/AI_ASSISTANCE.md`, `environments/` lock.
- Blender: `completion_operator.py` (sidecar `numerical_config.json` selects the finite-support backend), bundled `_vendor/neuro3d_completion`, `run_completion_selftest.py`, `check_installed_extension.py`; add-on version 0.2.0.
- CI workflow `completion.yml` (tests, smoke run, artifacts; Python 3.10, 3.12, 3.13).
- `experiments/results/contrato-008/`: the interrupted CONTRATO-008 run of 2026-10-10, archived unchanged with checksums, and `experiments/results/mnist_fashion_summary.json`.

### Changed
- `blender/neuro3d_lab/operators.py`: `OPTLAB_OT_run_bench.execute` is the versioned callback (legacy behaviour without the sidecar, identical to 7.8e-16).
- `blender/selftest/run_selftest.py`: exits 0 only if every step is PASS (a PENDING step no longer counts as success).
- README, ROADMAP (status update section added at the top), CITATION (version 0.2.0-dev): current results, limitations and instructions.

### Findings recorded
- One numerical sample per 0.533 λ pixel is not a converged model of a pixelated element (scores change 270 % from r = 1 to 2, 1.45 % from 8 to 16; observed order 2.05).
- Padding 4 with an absorbing aperture is adequate for scores and labels, not for fields.
- CONTRATO-008 (periodic model, interrupted): D2NN 0.933 / 0.825 on MNIST / Fashion-MNIST against a CNN at 0.979 / 0.892.

## [0.1.1] — 2026-10-10
- Blender extension package with bundled optlab submitted to extensions.blender.org (awaiting review).

## [0.1.0] — 2026-10-09 to 2026-10-10
- optlab core, Blender add-on, CONTRATO-001 to CONTRATO-011 (see `docs/CONTRACTS/`).
