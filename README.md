<p align="center">
  <img src="docs/img/header.gif" alt="Neuro3D-Lab: Gaussian-beam propagation, numerical width compared with the analytic width" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="Core licence: Apache-2.0" src="https://img.shields.io/badge/core%20licence-Apache--2.0-22d3ee"></a>
  <a href="blender/neuro3d_lab/LICENSE"><img alt="Blender add-on licence: GPL-3.0-or-later" src="https://img.shields.io/badge/add--on%20licence-GPL--3.0--or--later-a78bfa"></a>
  <img alt="Blender 4.5.14" src="https://img.shields.io/badge/Blender-4.5.14-fbbf24">
  <img alt="Tests: 98 passing" src="https://img.shields.io/badge/tests-98%20passing-34d399">
  <img alt="Status: simulation only" src="https://img.shields.io/badge/status-simulation%20only-f87171">
</p>

<p align="center"><b>A virtual optical bench. Blender lays out the bench and shows the result; a tested Python core computes the scalar field.</b></p>

---

## Blender Extensions

Version 0.2.1 was uploaded to Blender Extensions on 2026-10-13 in response to the review of 0.1.1: [review status](https://extensions.blender.org/approval-queue/neuro3d-lab/), [version history](https://extensions.blender.org/add-ons/neuro3d-lab/versions/#v021). The extension now uses relative imports for both bundled backends and leaves `sys.path` unchanged. The website description and release notes are in English, with a Spanish summary. Clean-profile installation, both simulation paths and 100 repository tests pass (`evidence/blender-review-20261013/`). Approval is pending.

## Abstract

Neuro3D-Lab is a simulation project for coherent scalar optical fields. It has three parts:

- **optlab** (`src/optlab`), the original NumPy core. It propagates a scalar field with the angular-spectrum method on a periodic FFT window and models Fresnel interfaces, a Mach–Zehnder interferometer and a phase-only diffractive neural network (D2NN).
- **neuro3d_completion** (`src/neuro3d_completion`), a versioned finite-support backend added in October 2026. It separates the physical phase support, the numerical quadrature (tied subpixels) and the FFT window, adds an absorbing aperture, direct Rayleigh–Sommerfeld references, convergence estimates, a Torch implementation for GPUs, and a frozen benchmark protocol with a TEST ledger. It reproduces optlab exactly where both apply.
- **A Blender add-on** (`blender/neuro3d_lab`) that builds the bench, exports it, runs either backend and displays the detector intensity.

The project is **not** a physical device. It contains no measured data. Each quantitative claim is linked to a contract with thresholds fixed before the run, or is labelled exploratory or post hoc. Failures are published with their values.

**Main findings so far.** (1) On every test set used so far, the phase-only D2NN does **not** beat the digital baselines. On the fresh writer-disjoint digits set (CONTRATO-007) it reaches 95.1 % against 96.1 % for a small CNN (McNemar p = 0.006). On MNIST and Fashion-MNIST, an interrupted run under the old periodic model (CONTRATO-008) gave 93.3 % and 82.5 %, against 97.9 % and 89.2 % for a CNN. (2) The periodic FFT window used by every historical D2NN result is **material**: it changes DEV accuracy from 0.955 to 0.850–0.866 (CONTRATO-010, CONTRATO-011). (3) With the 0.4 mm = 0.533 λ pixel, one numerical sample per pixel (the sampling used by the historical results and by part of the literature) is **not** a converged model of a pixelated phase element. Refining each pixel into tied subpixels changes detector scores by 270 % from r = 1 to 2, 6 % from 4 to 8 and 1.45 % from 8 to 16, with observed order 2.05 (CONTRATO-012). (4) Large pre-registered benchmarks with a fresh QMNIST50K test (CONTRATO-013) and a study of what this sampling does to trained masks (CONTRATO-014) are **running**. A 2D Maxwell check of the thin-element model (CONTRATO-015) is pre-registered and set up; it waits for the workstation's Linux environment.

---

## What changed in October 2026 (completion protocol)

The project was audited and extended following a written completion guide. Its code was integrated additively: no historical file or record was removed, and the historical contracts keep their verdicts. In summary:

| Area | What was done | Where |
|---|---|---|
| Finite-support model | Physical support, tied-subpixel quadrature and FFT window as three separate scales; `periodic` / `transparent` / `aperture` exteriors; exact adjoint gradients; Torch mirror for CUDA | `src/neuro3d_completion/`, [docs/API.md](docs/API.md) |
| Compatibility | New backend equals optlab (periodic, p = 1, r = 1) to 3e-15 in field, scores and gradient; NumPy and Torch agree to 1e-15 on CPU and CUDA | `evidence/legacy-compatibility.json`, `evidence/gpu-probe-rtx3090.json` |
| Reproduction of history | CONTRATO-002, the negative control, CONTRATO-010 and CONTRATO-011 re-run at commit e5ebb98 in a separate checkout: every number identical | `evidence/phaseA-historical-replay.json` |
| Numerical closure | Window and quadrature convergence, direct Rayleigh–Sommerfeld for a pixelated screen and a whole network, observed order and GCI, cross-check of two independent implementations | [CONTRATO-012](docs/CONTRACTS/CONTRATO-012-numerical-closure.md) |
| Data | MNIST, Fashion-MNIST and QMNIST admitted by SHA-256; MNIST split by NIST writer (0 shared writers); QMNIST50K kept as the only fresh test | `data/*/manifest.json` |
| Pre-registration | CONTRATO-013, 014 and 015 committed and pushed before any run, with hash manifests; a separate reviewer agent checked them first | `docs/CONTRACTS/`, `evidence/protocol-manifest-*.json` |
| Statistics | Exact McNemar power and minimum detectable effects; writer-grouped bootstrap; design effect 4.4–5.8 measured on DEV | `evidence/power-design-*.json`, `evidence/v2-review/writers.json` |
| History archived | The interrupted CONTRATO-008 run (MNIST / Fashion-MNIST, 2026-10-10) archived with checksums | `experiments/results/contrato-008/` |
| Blender | Versioned Run Bench path (finite-support backend through a `numerical_config.json` sidecar), bundled NumPy backend with an identity test, strict self-test exit code, clean-profile install check | `blender/`, `evidence/blender-final/` |
| Prior art | Registered search on sampling per sub-wavelength pixel; no novelty claimed for the tied-subpixel construction | [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) |
| Tooling | Sharded DEV selection (identical to the single-process version, tested), frozen-code checks, fidelity study, resumable queue driver, 4 runnable examples | `scripts/completion_tools/`, `examples/` |
| Disclosure | Which AI assistant did what | [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md) |

### Numerical closure in numbers (CONTRATO-012)

| Question | Result | Verdict against the pre-set criterion |
|---|---|---|
| Does the window size matter? (padding p → 2p, refinement 1, trained historical masks) | p 4 → 8: scores 0.15 %, no label changes, but field 2.6 %; p 8 → 16: field 0.6 % | Scores and labels pass at p = 4; fields need p = 8 |
| Is one sample per 0.533 λ pixel a converged model of a pixelated element? | Scores change 270 % (r 1 → 2), 28 % (2 → 4), 6 % (4 → 8), 1.45 % (8 → 16) | **Fail**; observed order 2.05, remaining error at r = 16 ≈ 0.5 % |
| Does the FFT model agree with direct Rayleigh–Sommerfeld? | Smooth Gaussian: 2e-8. Pixelated random screen: r = 1 stalls at 4.8 %, r = 3 reaches 0.8 %. Three-layer random network: r = 1 differs by 44–52 % in scores | Pass for smooth fields and r ≥ 3; **fail** for r = 1 |
| Do the two independent implementations agree? | The remaining 0.3 % is the evanescent field, which the model omits by design; adding it gives 0.045 % | Pass |

Consequence: every D2NN number in this repository, historical or new, describes the **sampled scalar model** (one sample per pixel). It is not a prediction for a fabricated pixelated element. CONTRATO-014 measures what happens to trained masks when the same pixels are simulated more faithfully.

### Pre-registered studies in progress

| Contract | Question | Status |
|---|---|---|
| [CONTRATO-013](docs/CONTRACTS/CONTRATO-013-closure-benchmark.md) | Writer-disjoint MNIST, Fashion-MNIST and fresh QMNIST50K. D2NN (aperture, p = 4, r = 1) against linear, MLP-12, MLP-128, CNN and a parameter-matched quadratic model, with three controls, 10 seeds each and paired statistics | Registered at `9521a9b`; DEV selection running on the shared GPU |
| [CONTRATO-014](docs/CONTRACTS/CONTRATO-014-fidelity.md) | Do masks trained with one sample per pixel keep their accuracy at refinement 2–16 and under direct RS? | Registered at `9521a9b`; 9 of 15 training runs done |
| [CONTRATO-015](docs/CONTRACTS/CONTRATO-015-maxwell.md) | 2D Maxwell (Meep FDTD) check of the thin-element model: vacuum, Fabry–Perot slab, discretisation, two-pixel element | Registered at `4e8e17f`; Meep 1.35.0 installed; waiting for WSL to restart |

---

## Scope of evidence

| Claim | Evidence | Artifact | Status |
|---|---|---|---|
| The scalar angular-spectrum propagator reproduces an analytic Gaussian beam, including its Gouy phase | CONTRATO-002, E1: worst relative error 4.1 × 10⁻⁵ (threshold 5 × 10⁻³) | `experiments/run_validation.py`, `experiments/results/validation_summary.json` | Pass, pre-registered; reproduced 2026-10-12 |
| Talbot self-imaging is reproduced at the Talbot distance | CONTRATO-002, E2: worst relative error 2.1 × 10⁻⁶ (threshold 1 × 10⁻²) | same as above | Pass, pre-registered |
| On-axis Fresnel diffraction by a circular aperture matches the analytic formula | CONTRATO-002, E3: worst error 7.2 × 10⁻³ (threshold 5 × 10⁻²) | same as above | Pass, pre-registered |
| Analytic tilted plane waves propagate correctly on the D2NN grid | CONTRATO-004, T1: maximum relative error 3.7 × 10⁻¹⁴ (threshold 1 × 10⁻¹⁰) | `experiments/run_regime_d2nn.py` | Pass, pre-registered |
| The D2NN-grid intensity does not depend on zero padding (historical periodic model) | CONTRATO-004, T2: 4.4 × 10⁻² (threshold 2 × 10⁻²) | same as above | Fail, published; mechanism later identified as the periodic window (CONTRATO-006c, CONTRATO-010) |
| The D2NN-grid result is converged in pixel size | CONTRATO-004, T3: 3.04 (threshold 5 × 10⁻²); CONTRATO-012 quantifies the refinement series | same, and `evidence/pixel-historical-p2.json`, `evidence/refinement-4-8-16-historical.json` | Fail; r ≥ 8 needed for 2 % in scores |
| A reversed propagation sign is detected by the Gouy-phase check | Gouy-phase errors of 0.49 and 1.57 rad | `experiments/negative_control.py`, `experiments/results/negative_control.json` | Control, not pre-registered; reproduced 2026-10-12 |
| Core and backend functions behave as specified | 98 tests pass (33 historic, 34 + 22 from the completion kit, 9 for the completion tools and examples), on Windows (Python 3.13) and in CI on Linux (Python 3.10, 3.12, 3.13) | `tests/`, `.github/workflows/` | Unit and integration tests |
| The add-on builds, exports, simulates and imports a bench | Self-test 15/15; new finite-support path 8/8 (identity with the legacy simulation 7.8 × 10⁻¹⁶; deliberate failure rejected); installed 0.2.0 package in a clean profile reproduces the repository run exactly | `blender/selftest/`, `evidence/blender-final/` | Pass; run with a script, not in CI |
| The D2NN classifies 8 × 8 digits above the pre-registered thresholds | CONTRATO-003, H1: TEST mean 0.962; controls pass | `experiments/results/d2nn_digits_summary.json` | Pass, pre-registered; historical periodic sampled model |
| The D2NN beats a digital baseline | CONTRATO-003 H2 (p = 0.22), CONTRATO-007 (CNN better, p = 0.006), CONTRATO-008 (CNN better on MNIST and Fashion) | contracts and `experiments/results/` | **Not supported** so far; CONTRATO-013 running |
| Any result applies to a physical device or a measurement | None | none | Not claimed; simulation only, no hardware experiment exists |

---

## Pipeline and the Blender bench

<p align="center">
  <img src="docs/img/pipeline.gif" alt="Pipeline: Blender exports the bench, optlab computes the field, and the result returns to Blender" width="85%">
</p>

Blender exports the bench geometry and the phase heights (`optlab.bench/1`). The core propagates the field and returns the detector intensity, which Blender displays as an emissive image. Blender only visualises; it does not compute the field. Without a sidecar file the add-on runs the original optlab simulation; with `numerical_config.json` (`padding`, `refinement`, `boundary`, `filter_mode`) it runs the finite-support backend and stores the settings, hashes and scores next to the result.

<p align="center">
  <img src="docs/img/blender_bench.gif" alt="Optical bench rendered in Blender: source, two phase layers and a detector showing the calculated intensity" width="90%">
</p>

The bench demo uses N = 256 pixels with an 8 µm pixel, λ = 633 nm, and planes at 0, 20, 40 and 60 mm. It is rendered with Cycles on the CPU. For visibility only, each plane is enlarged 15 times in x and y, so the image is a schematic and not a to-scale picture. It illustrates simulated output; it is not a measured result.

---

## Physics implemented

### Angular-spectrum propagation

Free-space propagation with the scalar angular-spectrum method. The transfer function applies to propagating components; evanescent components are not propagated. The formula contains no paraxial approximation.

$$U = \mathcal{F}^{-1}\{H\cdot\mathcal{F}\{u_0\}\},\qquad H = \exp\!\left(i\,2\pi d\sqrt{1/\lambda^2 - f_x^2 - f_y^2}\right)$$

In optlab the plain periodic method is valid only for $d < N\,dx^2/\lambda$; the simulator refuses out-of-domain segments unless `allow_out_of_domain=True` is passed. The finite-support backend instead enlarges the FFT window (padding) around a fixed physical support and measures the convergence of that choice. The band-limited method of Matsushima and Shimobaba (2009) is still not implemented; the backend offers only an explicit conservative `phase_gradient` filter, which is not that algorithm.

<p align="center">
  <img src="docs/img/propagation.gif" alt="Gaussian beam propagating; the numerical width matches the analytic width" width="90%">
</p>

- **CONTRATO-002, E1:** the width and Gouy phase of a Gaussian beam match the analytic solution, with worst relative error 4.1 × 10⁻⁵ (threshold 5 × 10⁻³). The Gaussian reference is a paraxial solution.
- **CONTRATO-004, T1:** analytic tilted plane waves on the D2NN grid match the analytic solution, with maximum relative error 3.7 × 10⁻¹⁴ (threshold 1 × 10⁻¹⁰).
- **CONTRATO-012:** against direct Rayleigh–Sommerfeld summation (an independent quadrature, no FFT), a smooth Gaussian agrees to 2 × 10⁻⁸ (n = 32).

### Fresnel interfaces

Reflectance and transmittance for normal and oblique incidence, including the Brewster angle and total internal reflection. R + T = 1 holds by construction, so it is not independent evidence. The completion backend adds coefficients that keep the evanescent transmission amplitude in total internal reflection while the transmitted flux is zero. Unit tests cover these functions; there is no analytic contract for them.

<p align="center">
  <img src="docs/img/fresnel.gif" alt="Fresnel reflectance for air-glass with the Brewster angle" width="90%">
</p>

### Mach–Zehnder interferometer

A unitary matrix made of two 50/50 beam splitters and a phase shifter. With the input at port 1, the output power follows sin²(θ/2). Unitarity is checked at sampled phase values in the unit tests, not over the full range.

<p align="center">
  <img src="docs/img/mzi.gif" alt="Power split of a Mach-Zehnder interferometer as the phase changes" width="90%">
</p>

### Talbot self-imaging

An amplitude grating with period p reproduces its image without a lens at $z_T = 2p^2/\lambda$. At $z_T/2$ the pattern differs from the input. The reference for this effect is paraxial.

<p align="center">
  <img src="docs/img/talbot.gif" alt="Talbot self-image: the intensity profile repeats at z_T" width="90%">
</p>

- **CONTRATO-002, E2:** the relative error at $z_T$ is 2.1 × 10⁻⁶ (threshold 1 × 10⁻²).

### Fresnel diffraction by a circular aperture

On-axis intensity compared with the Fresnel solution $I/I_0 = 4\sin^2(\pi N_F/2)$. The radius used is the effective radius of the sampled aperture, not the nominal radius; the contract declares this.

<p align="center">
  <img src="docs/img/aperture.gif" alt="On-axis intensity of a circular aperture, numerical and analytic" width="90%">
</p>

- **CONTRATO-002, E3:** the worst error is 7.2 × 10⁻³ (threshold 5 × 10⁻²).

---

## Diffractive neural network (D2NN)

The D2NN is a scalar, phase-only model. It has three phase layers, 18 mm between planes, a 128 × 128 pixel support with pixel pitch 0.4 mm (49,152 phases), and a detector of ten regions. The wavelength is λ = 0.75 mm (about 0.40 THz, as in Lin et al. 2018), so the pixel is 0.533 λ, slightly coarser than λ/2.

<p align="center">
  <img src="docs/img/d2nn_flow.gif" alt="Flow of a digit through the network: the energy is shared among 10 detector regions" width="95%">
</p>

The animation uses random phases to show how light flows through the network. It is not a trained result.

Two numerical models exist in the repository:

| Model | Window | Pixel sampling | Used by |
|---|---|---|---|
| Historical (optlab) | periodic 51.2 mm FFT window | one sample per pixel | CONTRATO-003 to CONTRATO-011 |
| Finite support (neuro3d_completion) | absorbing aperture, padding p (p = 4: 204.8 mm window) | refinement r tied subpixels per pixel (r = 1 for the benchmark) | CONTRATO-012 to CONTRATO-015 |

### Sampling and boundary checks

- **CONTRATO-004, T1 (pass):** analytic tilted plane waves, as above.
- **CONTRATO-004, T2 (fail):** N = 128 against N = 256 at the same pitch differs by 4.4 × 10⁻². CONTRATO-006c later showed that the periodic images of the FFT window explain the CONTRATO-006 V2 discrepancy, and CONTRATO-010/011 showed that the window changes the trained network's DEV accuracy from 0.955 to 0.850 (zero padding, transparent exterior) or 0.866 (absorbing aperture).
- **CONTRATO-004, T3 (fail as pre-registered):** pixel convergence, 3.04. CONTRATO-012 measured the full refinement series (see above): one sample per pixel is not converged.

---

## Scientific method: contracts and verdicts

Each contract states its thresholds before its run. The git history records when each threshold was committed. Since October 2026, new contracts are also pushed to GitHub before their runs, together with hash manifests. That gives a server-side timestamp, but it is still not an external registry (OSF or Zenodo), which is pending. Failures are published with their values; post hoc analyses are labelled as such.

| Contract | What it tests | Status |
|---|---|---|
| [CONTRATO-001](docs/CONTRACTS/CONTRATO-001-nucleo.md) | Core unit tests and the Blender self-test | Pass (33 historic tests; self-test 15/15) |
| [CONTRATO-002](docs/CONTRACTS/CONTRATO-002-validacion-analitica.md) | Analytic validation of the propagator | E1, E2 and E3 pass; E4 exploratory |
| [CONTRATO-003](docs/CONTRACTS/CONTRATO-003-d2nn-digitos.md) | D2NN on 8 × 8 digits, with controls | H1 pass, H2 not met, controls pass |
| [CONTRATO-004](docs/CONTRACTS/CONTRATO-004-regimen-d2nn.md) | D2NN sampling regime (dx/λ = 0.53) | T1 pass; T2 and T3 fail |
| [CONTRATO-005](docs/CONTRACTS/CONTRATO-005-robustez-n256.md) | N = 256 robustness (post hoc) | Seed-0 TEST 0.9639 → 0.9694; stable |
| [CONTRATO-006](docs/CONTRACTS/CONTRATO-006-rayleigh-sommerfeld.md), [006b](docs/CONTRACTS/CONTRATO-006b-diagnostic.md), [006c](docs/CONTRACTS/CONTRATO-006c-periodic-sampling.md) | Rayleigh–Sommerfeld reference and diagnostics | V2 fail; periodic window explains it (post hoc) |
| [CONTRATO-007](docs/CONTRACTS/CONTRATO-007-test-fresco.md) | Fresh writer-disjoint digits test, 10 seeds | D2NN 0.9505; CNN 0.9613; H2 fail (p = 0.006) |
| [CONTRATO-008](docs/CONTRACTS/CONTRATO-008-mnist-fashion.md) | MNIST and Fashion-MNIST (periodic model) | Interrupted and incomplete; archived results: D2NN 0.933 / 0.825 vs CNN 0.979 / 0.892 (seed 0) |
| [CONTRATO-009](docs/CONTRACTS/CONTRATO-009-pytorch-replication.md) | Independent PyTorch replication | Pass (≤ 6 × 10⁻¹⁴) |
| [CONTRATO-010](docs/CONTRACTS/CONTRATO-010-c12-periodic-dev.md) | Periodic window on DEV (post hoc) | Material (0.955 → 0.850) |
| [CONTRATO-011](docs/CONTRACTS/CONTRATO-011-aperture-dev.md) | Aperture convention on DEV (post hoc) | R3, inconclusive (0.866) |
| [CONTRATO-012](docs/CONTRACTS/CONTRATO-012-numerical-closure.md) | Numerical closure (window, quadrature, RS, GCI) | Record: see the table above |
| [CONTRATO-013](docs/CONTRACTS/CONTRATO-013-closure-benchmark.md) | Closure benchmark with fresh QMNIST50K | Pre-registered; running |
| [CONTRATO-014](docs/CONTRACTS/CONTRATO-014-fidelity.md) | Discretisation fidelity of trained masks | Pre-registered; running |
| [CONTRATO-015](docs/CONTRACTS/CONTRATO-015-maxwell.md) | 2D Maxwell check of the thin-element model | Pre-registered; waiting for its environment |

**Validation summary**

| Check | Quantity | Result | Threshold | Verdict |
|---|---|---|---|---|
| E1, Gaussian beam width and Gouy phase (CONTRATO-002) | worst relative error | 4.1 × 10⁻⁵ | 5 × 10⁻³ | Pass |
| E2, Talbot self-imaging at $z_T$ (CONTRATO-002) | worst relative error | 2.1 × 10⁻⁶ | 1 × 10⁻² | Pass |
| E3, on-axis circular aperture (CONTRATO-002) | worst error in $I/I_0$ | 7.2 × 10⁻³ | 5 × 10⁻² | Pass |
| E4, smooth Gaussian beyond the validity limit (CONTRATO-002) | not applicable | not applicable | not applicable | Exploratory, no verdict |
| Negative control: reversed propagation sign (CONTRATO-002) | Gouy-phase error | 0.49 and 1.57 rad | not applicable | Detected; control, not pre-registered |
| T1, tilted plane waves (CONTRATO-004) | maximum relative error | 3.7 × 10⁻¹⁴ | 1 × 10⁻¹⁰ | Pass |
| T2, zero padding N = 128 against N = 256 (CONTRATO-004) | relative L2 difference in intensity | 4.4 × 10⁻² | 2 × 10⁻² | Fail |
| T3, pixel convergence (CONTRATO-004) | relative L2 difference in intensity | 3.04 | 5 × 10⁻² | Fail as pre-registered |
| New backend vs optlab, periodic (CONTRATO-012) | field / gradient | 1.7 × 10⁻¹⁵ / 3.2 × 10⁻¹⁵ | 1 × 10⁻⁹ / 1 × 10⁻⁸ | Pass |
| Window p 4 → 8, DEV inputs (CONTRATO-012) | scores / field | 1.5 × 10⁻³ / 2.6 × 10⁻² | 1 × 10⁻² each | Scores pass; field fail |
| Refinement r 8 → 16 (CONTRATO-012) | scores | 1.45 × 10⁻² | 1 × 10⁻² | Fail (GCI at r = 16: 0.58 %) |

<p align="center">
  <img src="docs/img/verdicts.gif" alt="Verdicts for each pre-registered check" width="90%">
</p>

---

## Digit classification (CONTRATO-003)

A three-layer, phase-only D2NN was trained on handwritten digits (scikit-learn `load_digits`, 8 × 8 images upsampled to 64 × 64). It was evaluated once on TEST (360 images, five seeds). The thresholds were fixed in advance, and the configuration was chosen on DEV only. Historical periodic sampled model.

| Code | Criterion | TEST result | Verdict |
|---|---|---|---|
| H1 (primary) | Mean over 5 seeds ≥ 0.85, and each seed ≥ 0.80 | mean 0.962, minimum 0.958 | Pass |
| H2 | D2NN beats a logistic regression on the same input (paired difference ≥ 0 and McNemar p < 0.05) | +0.009 against 0.953; p = 0.22 | Fail |
| C1 | Permuted training labels: mean ≤ 0.15 | 0.126 | Pass |
| C2 | Untrained random phases: mean ≤ 0.20 | 0.124 | Pass |
| C3 | No phase layers: ≤ 0.20 | 0.025 | Pass |

The D2NN classifies well above chance and above the H1 threshold. It does **not** beat the linear baseline significantly. The full registry, the per-seed intervals and the confusion-matrix diagonal are in [CONTRATO-003](docs/CONTRACTS/CONTRATO-003-d2nn-digitos.md).

<p align="center">
  <img src="docs/img/d2nn_results.gif" alt="Handwritten digits on TEST: D2NN seeds, their mean, the logistic baselines and the three controls, against the 0.85 threshold" width="90%">
</p>

---

## Limitations

- **Scalar thin-element model.** There is no polarisation and no vector field, and each phase layer is an infinitely thin phase screen. At a 0.533 λ pixel with relief heights of the order of λ, this may not hold. The 2D Maxwell check (CONTRATO-015) is pre-registered and set up but has not run yet, so this limitation is still open.
- **Pixel sampling.** All D2NN accuracies, historical and in CONTRATO-013, use one numerical sample per 0.533 λ pixel. CONTRATO-012 shows that this is not a converged model of a piecewise-constant element (r ≥ 8 is needed for scores within 2 %). Read every accuracy as a property of the sampled model. CONTRATO-014 measures the effect on trained masks.
- **Window.** Historical results use a periodic window that changes the trained network's behaviour (CONTRATO-010). The new backend uses an absorbing aperture with padding 4, which is converged for scores and labels (0.15 %) but not for fields (2.6 %; padding 8 is needed).
- **Evanescent waves, reflections and dispersion.** Evanescent components are omitted by design (about 0.3 % at 8 λ in a cross-check). Plane-to-plane reflections and spectral dispersion are not modelled.
- **Band-limited propagation.** The Matsushima–Shimobaba band-limited method is not implemented. The optional `phase_gradient` filter is a different, explicit construction.
- **No certified error bounds.** The contraction budget and margin certificate are implemented, but no operator-error bound has been proved, so no label is certified. Convergence numbers are estimates (GCI), not bounds.
- **Relative intensity and ideal detection.** D2NN results are energy fractions, not watts. Noise models (photon counting, NEP) exist for exploratory curves only; no detector is calibrated.
- **Simulation only.** No physical hardware exists and no measured data are used. Nothing here supports a claim about optical speed or energy.
- **Re-used test sets.** The official MNIST and Fashion-MNIST test sets were examined by CONTRATO-008. Only QMNIST50K is fresh, and only for MNIST. Inference to new writers has a design effect of about 5, so the detectable differences are larger than for independent images.
- **External verification.** Pre-registrations were checked by a separate AI agent, and CI runs the tests on Linux. No human external reviewer has reproduced the results, no external registry entry exists, and no DOI has been assigned.
- **Blender self-test.** It is run with scripts and logged, but it is not part of continuous integration.

---

## Installation and reproduction

Requirements: Python 3.10 or newer (CI checks 3.10, 3.12 and 3.13). Blender 4.5.14 is needed only for the add-on checks. PyTorch is needed only for the benchmark, the fidelity study and the Torch tests; Meep (conda-forge `pymeep`, Linux or macOS) only for CONTRATO-015.

Linux or macOS shell:

```bash
python -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[dev,experiments,figures]" -r requirements-completion.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
python scripts/check_legacy_compatibility.py --output legacy-compatibility.json
python scripts/smoke_run.py --output completion-smoke
```

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,experiments,figures]" -r requirements-completion.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
```

Examples (`examples/`): scalar reference against Rayleigh–Sommerfeld, a convergence study with observed order, inference with saved masks (refuses TEST files), and the calibrated-measurement analysis on explicitly synthetic data.

Historical analytic validation and experiments:

```bash
python experiments/run_validation.py         # CONTRATO-002
python experiments/run_regime_d2nn.py        # CONTRATO-004
python experiments/run_d2nn_digits.py        # CONTRATO-003, about 3.6 h on CPU
python docs/figures/make_figures.py          # README figures
```

Completion protocol (data, diagnostics, benchmark): admit the datasets by SHA-256, then run the phases of the contracts.

```bash
python -m neuro3d_completion.data --dataset mnist --directory datasets/mnist --output data/mnist
python -m neuro3d_completion.data --dataset fashion --directory datasets/fashion --output data/fashion
python -m neuro3d_completion.qmnist prepare --mnist datasets/mnist --qmnist datasets/qmnist --output data/mnist-writer
python -m neuro3d_completion.validation pixel --n 128 --phases experiments/results/d2nn_seed0_phases.npz --base-padding 2 --refinements 1 2 4 8 --output pixel.json
python scripts/completion_tools/make_jobs.py --out jobs
python scripts/completion_tools/queue_driver.py jobs/gpu_jobs.json --logs logs/gpu
```

`queue_driver.py` sends GPU jobs through the owner's workstation queue; set `NEURO3D_GPUQ` or run the job commands directly elsewhere. Blender add-on checks (replace `path/to/blender`):

```bash
python scripts/run_blender_checks.py --blender "path/to/blender" --repo . --output blender-checks
"path/to/blender" -b --factory-startup -P blender/selftest/run_completion_selftest.py
```

By default, historical outputs go to `.cognition/neuro3d-lab` in the folder above the repository; set `NEURO3D_LAB_OUT` to use another folder. Datasets and split files are not committed; their hashes are.

---

## Repository structure

```
src/optlab/                   original scalar-field core (periodic FFT window)
src/neuro3d_completion/       finite-support backend, benchmark protocol, references, statistics
blender/neuro3d_lab/          Blender add-on (GPL-3.0-or-later), bundled copies of both cores in _vendor/
blender/selftest/             headless self-tests (legacy path, new path, installed package)
experiments/                  scripts of CONTRATO-002 to CONTRATO-011
experiments/results/          versioned results; contrato-008/ holds the archived interrupted run
scripts/                      completion scripts (compatibility, smoke, release checks, report)
scripts/completion_tools/     sharded selection, fidelity study, verdict tools, queue driver, Maxwell check
completion-configs/           benchmark protocols (guide proposals and the registered p4/r1 versions)
data/                         split manifests with SHA-256 (data files are not committed)
evidence/                     records of every completion run, manifests, JEV decisions, Blender logs
examples/                     runnable examples
docs/                         model, architecture, API, contracts, literature review, AI-assistance note
environments/                 package locks
tests/                        unit and integration tests
.github/workflows/            CI: legacy tests and the completion protocol (Python 3.10, 3.12, 3.13)
```

---

## Related projects and prior art

No numerical comparison with these projects has been made yet: Blender Optics Simulator, BlenderPhotonics, TorchOptics, onn-simulation, POPPY. See [docs/RELATED_WORK.md](docs/RELATED_WORK.md) and, for the sampling question, [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md). In short, Mengu et al. (IEEE JSTQE 2020) sample one point per 0.53 λ neuron, as the historical model here does, and Luo et al. (Light Sci. Appl. 2019) oversample by 4 with tied subpixels. This project claims no novelty for tied-subpixel refinement.

---

## References

- Lin, X. et al. All-optical machine learning using diffractive deep neural networks. Science 361, 1004 (2018). [doi:10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084) · [arXiv:1804.08711](https://arxiv.org/abs/1804.08711)
- Matsushima, K. & Shimobaba, T. Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields. Optics Express 17, 19662 (2009). [doi:10.1364/OE.17.019662](https://doi.org/10.1364/OE.17.019662)
- Mengu, D., Luo, Y., Rivenson, Y. & Ozcan, A. Analysis of diffractive optical neural networks and their integration with electronic neural networks. IEEE JSTQE 26 (2020). [arXiv:1810.01916](https://arxiv.org/abs/1810.01916)
- Luo, Y. et al. Design of task-specific optical systems using broadband diffractive neural networks. Light Sci. Appl. 8, 112 (2019). [doi:10.1038/s41377-019-0223-1](https://doi.org/10.1038/s41377-019-0223-1)
- Kulce, O., Mengu, D., Rivenson, Y. & Ozcan, A. All-optical information-processing capacity of diffractive surfaces. Light Sci. Appl. 10, 25 (2021). [doi:10.1038/s41377-020-00439-9](https://doi.org/10.1038/s41377-020-00439-9)
- Yadav, C. & Bottou, L. Cold Case: The Lost MNIST Digits. NeurIPS 32 (2019). [arXiv:1905.10498](https://arxiv.org/abs/1905.10498)
- Celik, I. B. et al. Procedure for estimation and reporting of uncertainty due to discretization in CFD applications. J. Fluids Eng. 130, 078001 (2008). [doi:10.1115/1.2960953](https://doi.org/10.1115/1.2960953)
- Clements, W. R. et al. An optimal design for universal multiport interferometers. Optica (2016). [doi:10.1364/OPTICA.3.001460](https://doi.org/10.1364/OPTICA.3.001460)

Related work by the author: [Neuro3D](https://github.com/Agnuxo1/Neuro3D), [Silice-Neuro3D-Cube](https://github.com/Agnuxo1/Silice-Neuro3D-Cube), [NeuroPixel](https://github.com/Agnuxo1/NeuroPixel) and [OpenCLAW-P2P](https://github.com/Agnuxo1/OpenCLAW-P2P).

---

## Contributing, roadmap and changes

See [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md) and [CHANGELOG.md](CHANGELOG.md).

---

## How to cite

Use the metadata in [CITATION.cff](CITATION.cff). GitHub's "Cite this repository" button reads this file. No DOI has been assigned yet.

---

## Licence

- **optlab core** (`src/optlab`) and **neuro3d_completion** (`src/neuro3d_completion`): Apache License 2.0.
- **Blender add-on** (`blender/neuro3d_lab`): GNU General Public License v3.0 or later (GPL-3.0-or-later), with its own [LICENSE](blender/neuro3d_lab/LICENSE) file. The add-on uses GPL-3.0-or-later because the official Blender extensions catalogue requires it for add-ons. The bundled copies of the Apache-2.0 cores carry their LICENSE and NOTICE.

Repository licence text: [LICENSE](LICENSE) and [NOTICE](NOTICE). Blender itself is not redistributed in this repository.

Author: Francisco Angulo de Lafuente ([Agnuxo1](https://github.com/Agnuxo1)).
