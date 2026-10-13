# Roadmap and evidence plan

## Status update 2026-10-13 (completion protocol)

This section supersedes the status lines of the 2026-10-10 roadmap below, which is kept unchanged as a historical record. Status values: **done** (committed evidence), **in progress** (registered and running or partly done), **pending (external / owner)** (needs hardware, an external party or the owner's accounts). Totals over the 51 audit items: 27 done, 21 in progress, 2 pending (external), 1 pending (owner).

| ID | Status | Evidence or next action |
|---|---|---|
| G01 | in progress | RS references and operator-vs-RS done (CONTRATO-012); full-scale RS on trained masks running (CONTRATO-014) |
| G02 | done | Window convergence measured; aperture/transparent/periodic compared (CONTRATO-010/011/012); padding 4 adequate for scores and labels only |
| G03 | done | Tied-subpixel refinement 1-2-4-8-16, observed order 2.05, GCI (CONTRATO-012); historical T3 failure kept |
| G04 | in progress | CONTRATO-015 (2D Maxwell, Meep 1.35.0) pre-registered and installed; waiting for WSL |
| G05 | in progress | CONTRATO-013 (writer-disjoint MNIST, Fashion-MNIST, QMNIST50K) registered and running |
| G06 | in progress | Paired statistics, Holm, crossed and writer-grouped bootstrap implemented; results pending |
| G07 | in progress | Ten paired seeds for every model in CONTRATO-013; running |
| G08 | in progress | Linear, MLP-12, MLP-128, CNN and parameter-matched quadratic model in CONTRATO-013; running |
| G09 | pending (external) | No hardware or laboratory; analysis code and formats ready (hardware.py, example 4) |
| G10 | done | Every new threshold justified in its contract before the run (CONTRATO-013/014/015) |
| G11 | done | Independent-position finite differences, adjoint identities and autograd tests (tests/test_completion.py) |
| G12 | done | phase_gradient filter evaluated against direct RS as an explicit variant (CONTRATO-012) |
| G13 | in progress | CONTRATO-008 run archived; new runs keep weights, phases, predictions and logs |
| G14 | in progress | CI on Linux Python 3.10/3.12/3.13; Windows CUDA lock committed; Linux CPU lock and Meep export pending |
| G15 | in progress | Final runs save phases and checksums for every seed |
| G16 | in progress | Contracts pushed before runs with hash manifests; external registry (OSF/Zenodo) needs the owner's account |
| G17 | pending (external) | Third-party reproduction not yet requested |
| G18 | in progress | docs/LITERATURE_REVIEW.md (sampling prior art); full comparison pending |
| G19 | done | README updated with CONTRATO-010 to 015 and current limitations |
| G-01 | in progress | New documentation in English; historical Spanish file names and coordination notes kept |
| G-02 | done | scripts/audit_docs.py: no missing local links |
| G-03 | pending (owner) | Public metadata is the owner's decision |
| G-04 | done | No machine paths in new tools (NEURO3D_PYTHON, NEURO3D_GPUQ); historical scripts keep their documented paths |
| G-05 | in progress | Licences and README updated; single status source = this section |
| G-06 | done | Smoke run from a folder outside the repository; CI on Linux |
| G-07 | in progress | CI runs tests, smoke run and uploads artifacts on three Python versions; Blender not in CI |
| G-08 | in progress | Add-on 0.2.0; package version stays 0.1.0 until the closure gates |
| G-11 | done | Extension validate and clean-profile install of 0.2.0 (evidence/blender-final) |
| G-12 | done | Bundled neuro3d_completion with identity test (tests/test_vendor_sync_completion.py) |
| G-13 | done | Self-test exits non-zero unless every step is PASS |
| G-18 | done | Spectrum, window and quadrature measured jointly (CONTRATO-012) |
| G-19 | done | Fresnel/TIR coefficients with evanescent amplitude (neuro3d_completion.interfaces, tests) |
| G-20 | done | Shapes, labels, empty batches, NaN/inf and physical parameters validated (tests) |
| G-21 | in progress | Per-model intervals and Holm implemented; results pending |
| G-22 | in progress | CUDA lock committed; Meep environment export pending WSL; Blender version recorded |
| G-23 | in progress | Per-sample predictions and dataset hashes saved by every final run |
| G-24 | done | requirements-completion.txt; Torch not required inside Blender |
| G-25 | done | docs/API.md; py.typed not offered (no full static typing) |
| G-26 | done | examples/ (four runnable examples, tested) |
| G-27 | in progress | CITATION and CHANGELOG updated; release with evidence archive and DOI pending |
| C01 | in progress | Figures from the CONTRATO-013/014 results pending |
| C02 | done | Blender bench declared illustrative and scaled |
| C03 | done | Paraxial references are not called exact |
| C04 | done | Header states the verified scope only |
| C05 | done | Confirmatory, post hoc and exploratory labels kept |
| C06 | done | Shell and PowerShell instructions in README |
| C07 | done | Test counts from real runs: 33 historic, 98 total |
| C08 | done | Python 3.10, 3.12, 3.13 checked in CI |
| C10 | done | Blender designs, exports and displays; the core computes the field |
| C11 | done | lambda = 0.75 mm (about 0.40 THz) declared |
| C12 | in progress | Boundary and sampling quantified; retraining under the finite-support model running (CONTRATO-013/014) |

Next actions: finish CONTRATO-013 and CONTRATO-014 on the shared GPU and record their results without changing thresholds; run CONTRATO-015 once WSL is restarted; secondary DEV analyses (robustness, detection noise, per-pixel influence) on the final masks; manuscript draft from the real summaries; external registration, DOI and replication with the owner.

---

## Historical status (2026-10-10)

Status date: 2026-10-10. Checked against commit `a5c81fc`, with uncommitted working-tree changes noted where they affect a status.

## 1. Purpose

The goal of this project is publication-grade evidence. Each quantitative claim should be linked to a pre-registered contract in `docs/CONTRACTS/`, a script in the repository, and a committed result. This roadmap does not claim any absolute level of quality, novelty or impact. It records what is verified, what is still pending, and the order in which the pending work should run.

### 1.1 Current state

- **Scope.** Simulation only. No physical hardware experiment exists.
- **Licences.** The optlab core (`src/optlab`) is Apache-2.0. The Blender add-on (`blender/neuro3d_lab`) is GPL-3.0-or-later, with its own `LICENSE` file. The official Blender extensions catalogue requires GPL-3.0-or-later for add-ons, and the add-on imports optlab.
- **Unit tests.** 32 pass (`python -m pytest -q`).
- **Blender self-test.** Headless, Blender 4.5.14: 15 of 15 steps PASS. The log is not versioned in the repository.
- **Blender manifest.** `blender/neuro3d_lab/blender_manifest.toml` validates with `blender --command extension validate blender/neuro3d_lab`.
- **CONTRATO-002 (analytic validation, pre-registered).** E1 (Gaussian beam with Gouy phase), E2 (Talbot self-imaging) and E3 (circular aperture on axis) pass with margin. Worst relative errors are 4.1e-5, 2.1e-6 and 7.2e-3 against thresholds 5e-3, 1e-2 and 5e-2. E4 is exploratory and has no verdict. The reversed-sign control gives Gouy-phase errors of 0.49 and 1.57 rad; its script and output are not yet in the repository (see 2.4).
- **CONTRATO-004 (D2NN sampling regime, dx/lambda = 0.53; pre-registered and amended before running).**
  - T1, analytic tilted plane waves: pass (maximum error 3.7e-14; threshold 1e-10).
  - T2, zero padding, N = 128 against N = 256 at 18 mm: fail (relative L2 4.4e-2; threshold 2e-2). Periodic wrap-around changes the intensity by about 4 percent in the N = 128 grid.
  - T3, pixel convergence: fail as pre-registered (3.04 against 5e-2). The contract wording "block-averaged" was wrong. An energy-conserving block sum gives 2.8e-2. That value is exploratory and post hoc, not a verdict.
- **CONTRATO-003 (digits D2NN, 3 phase layers, 18 mm spacing, N = 128).** In progress. No TEST accuracy is stated here. Validation (DEV) accuracies are preliminary and are not results.
- **Validity domain.** The plain angular-spectrum propagator is valid for d < N * dx^2 / lambda. `src/optlab/bench.py` refuses out-of-domain segments unless `allow_out_of_domain=True` is passed.

## 2. Status of blocking (P0) and important (P1) items

Items keep the identifiers of the internal audit: G01 to G19, G-01 to G-27 and C01 to C12. Only P0 and P1 items are listed here; P2 and P3 items remain in the audit. The effort column is the audit's estimate.

Status values:

- **done**: the fix is committed, and a committed file proves it.
- **in progress**: part of the fix is committed, or the fix exists only in uncommitted working-tree changes, or a run is active.
- **open**: no fix in the repository yet.

Logs kept outside the repository are not counted as evidence. Totals: 51 items; 2 done, 25 in progress, 24 open.

### 2.1 Numerics, validation, statistics and baselines (G01 to G19)

| ID | Priority | Title | Status | Evidence | Effort (audit) |
|---|---|---|---|---|---|
| G01 | P0 | numerics / propagator regime | in progress | CONTRATO-004 T1 pass (commit `15ce4f0`). No independent Rayleigh-Sommerfeld reference yet. | S (2-4 days) |
| G02 | P0 | numerics / periodic boundary | in progress | CONTRATO-004 T2 fail (commit `15ce4f0`). No zero-padding or band-limited fix yet. | S (1-3 days, mostly CPU time) |
| G03 | P0 | numerics / convergence | in progress | CONTRATO-004 T3 fail as pre-registered (commit `15ce4f0`). No distance sweep yet. | M (1-2 weeks including compute) |
| G04 | P0 | physics model / vector effects | open | No vector or FDTD reference in the repository. Scalar limits are listed in `docs/MODEL.md` section 8. | L (2-4 weeks) |
| G05 | P0 | benchmark | open | Only the scikit-learn digits dataset (8x8 images) is used. No MNIST or Fashion-MNIST run. | L (3-6 weeks including batched implementation) |
| G06 | P0 | statistics / uncertainty | open | `experiments/run_d2nn_digits.py` line 226 applies a Wilson interval to the seed mean. No standard deviation over seeds. | XS (<1 day) |
| G07 | P0 | statistics / test power | open | The test set has 360 samples. The McNemar comparison uses seed 0 only. | S (folded into G05) |
| G08 | P0 | baselines | open | The logistic regression baseline is part of CONTRATO-003 (in progress). No parameter-matched digital baseline yet. | S-M (about 1 week) |
| G09 | P0 | experimental validation | open | No measured data. All results are simulation. | XL (months; requires lab access) |
| G10 | P1 | validation tolerances | open | CONTRATO-002 thresholds unchanged. No error-versus-dx or error-versus-N tables in `experiments/results/`. | S (2-3 days) |
| G11 | P1 | test suite quality | open | `tests/test_d2nn_gradients.py` line 16 still samples from the 200 largest gradient entries. | S (2-3 days) |
| G12 | P1 | propagator / long distances | open | Band-limited angular spectrum not implemented in `src/optlab/propagation.py`. | S-M (about 1 week) |
| G13 | P1 | reproducibility of reported numbers | open | `experiments/results/` holds two summary files only. The full `validation.json` is not versioned. | XS (<1 day) |
| G14 | P1 | reproducibility / software packaging | in progress | Extras `experiments` and `figures` declared and CI added (commit `a5c81fc`). Still open: D: path in `experiments/run_d2nn_digits.py` line 46, no lock file, CI runs pytest only. | S (2-3 days) |
| G15 | P1 | data availability | open | `.gitignore` excludes `*.npy` and `*.npz`. No phase masks are committed. | XS (<1 day) |
| G16 | P1 | pre-registration credibility | in progress | Git order is visible for CONTRATO-004 (`e756a2a`, then amendment `6ac64ae`, then run `15ce4f0`). No tag marks the CONTRATO-003 state before its TEST run. See 2.4, item 4. | XS (<1 day) |
| G17 | P1 | independent replication | open | No second implementation, external reviewer or second-machine run. | M (1-2 weeks) |
| G18 | P1 | literature and positioning | open | The README reference list still has three entries (Lin 2018, Matsushima 2009, Clements 2016). | S (2-3 days) |
| G19 | P1 | claims versus evidence | in progress | A scope-of-evidence table is in the uncommitted README. Not yet committed. | XS (<1 day) |

### 2.2 Documentation, packaging, Blender and reproducibility (G-01 to G-27)

| ID | Priority | Title | Status | Evidence | Effort (audit) |
|---|---|---|---|---|---|
| G-01 | P0 | documentation-english | in progress | `README.md` is in English in the uncommitted changes. Spanish text remains in `docs/CONTRACTS/` (CONTRATO-001 to 003), `experiments/`, `docs/figures/make_figures.py` and `blender/selftest/run_selftest.py`. | L (1-2 days) |
| G-02 | P0 | docs-broken-links | in progress | The uncommitted README no longer embeds `docs/img/d2nn_results.gif`, which does not exist. No link-check test yet. | S (<1 h) |
| G-03 | P0 | git-privacy | open | All 16 commits carry the owner's personal author address. A history rewrite needs the owner's approval. | S (<1 h) plus owner approval |
| G-04 | P0 | portability-hardcoded-paths | in progress | The Blender scripts and the CONTRATO-002 script read `NEURO3D_LAB_OUT` (commits `4960f1a` and `a5c81fc`). Remaining: `experiments/run_d2nn_digits.py` lines 10 and 46 use D: paths; the committed README line 153 has a D: path (the uncommitted README uses a placeholder). | M (1-4 h) |
| G-05 | P0 | contract-and-history-consistency | open | `coordinacion/DECISIONES.md` line 7 still says the repository is private and unlicensed. `coordinacion/PROTOCOLO.md` line 5 refers to `CHECKPOINT.md`, which is not in the repository. | S-M (1-3 h) |
| G-06 | P1 | portability-bug | in progress | Output defaults no longer write to a relative D: folder in the committed validation and Blender scripts. `experiments/run_d2nn_digits.py` line 46 still writes to a D: path. | S (<1 h) |
| G-07 | P1 | ci | in progress | `.github/workflows/tests.yml` runs `pytest -q` on Python 3.13 (commit `a5c81fc`). No ruff, mypy or Blender job. No recorded run in the repository. | M (2-4 h) |
| G-08 | P1 | packaging-metadata | in progress | `pyproject.toml` has readme, authors, keywords, project URLs and license-files. No classifiers. | S (<1 h) |
| G-11 | P1 | blender-extension-manifest | done | `blender/neuro3d_lab/blender_manifest.toml` is committed (commit `a5c81fc`). The manifest check runs locally, not in CI. | S-M (1-3 h) |
| G-12 | P1 | blender-self-contained | open | `blender/neuro3d_lab/operators.py` lines 78-80 still insert an external optlab path into `sys.path`. | M (1-4 h) |
| G-13 | P1 | selftest-silent-pass | open | `blender/selftest/run_selftest.py` line 115 returns the string "PENDIENTE" when optlab is missing. The exit code (line 280) fails only on "FAIL". | S (<1 h) |
| G-18 | P1 | physics-sampling-validity | in progress | Domain check on propagation length committed (commit `ddb60ac`). No warning for dx above lambda/2 in `src/optlab/propagation.py`. CONTRATO-004 measures this regime. | M-L (1-2 days including the new contract) |
| G-19 | P1 | physics-input-validation | open | `src/optlab/fresnel.py` line 23 checks polarisation only. Incidence angles above 90 degrees are accepted. | S (<1 h) |
| G-20 | P1 | api-input-validation-d2nn | open | `src/optlab/d2nn.py` validates distances and mask shape only. No checks for labels, batches or input shape. | S (<1 h) |
| G-21 | P1 | statistics-contract-compliance | open | No standard deviation over seeds in `experiments/` or `docs/figures/`. The Wilson interval is applied to the seed mean. | S-M (1-3 h) |
| G-22 | P1 | reproducibility-environment | open | No lock file (`uv.lock`, requirements file or `environment.yml`) in the repository. | M (1-4 h) |
| G-23 | P1 | reproducibility-data | open | `experiments/results/` holds summaries only. The full `validation.json` and the per-sample D2NN predictions are not versioned. | S-M (1-3 h) |
| G-24 | P1 | packaging-dependencies | done | `pyproject.toml` declares the `experiments` and `figures` extras (commit `a5c81fc`). | S (<1 h) |
| G-25 | P1 | api-docs-typing | open | No `py.typed` marker in `src/optlab`. | M-L (1-2 days) |
| G-26 | P1 | examples-tutorials | open | No `examples/` directory. | M (1-4 h) |
| G-27 | P1 | project-metadata | in progress | `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md` and `CODE_OF_CONDUCT.md` are committed (commit `a5c81fc`). No `CHANGELOG.md`. | S-M (1-3 h) |

### 2.3 Claims and README (C01 to C12)

| ID | Priority | Title | Status | Evidence | Effort (audit) |
|---|---|---|---|---|---|
| C01 | P0 | README figures and links | in progress | Same missing-image reference as G-02. Removed in the uncommitted README. | small (minutes) |
| C02 | P0 | numerical validity of the Blender bench | in progress | Domain check (`ddb60ac`), in-domain demo with 256 pixels and 8 micrometre pitch, and re-rendered GIF (`4960f1a`) are committed. The uncommitted README states the enlargement as 15 times, which matches the code. The committed README still says 60 times. | medium (re-render with Blender, about 1 h) |
| C03 | P0 | physical validation claims (non-paraxial) | in progress | The uncommitted README removes the "exact" wording and states that the Gaussian reference is paraxial. CONTRATO-004 T1 (commit `15ce4f0`) tests analytic tilted waves. Remaining: convergence in dx and a high-angle reference (see G01). | medium (contract and test, CPU) |
| C04 | P0 | headline and header wording | in progress | The uncommitted README headline describes "a tested Python core". The committed README describes the core as verified. | small |
| C05 | P1 | method: pre-registration | in progress | The uncommitted README states that the git history records threshold order and that this is not an external pre-registration. Git order is visible for CONTRATO-004 (`e756a2a`, `6ac64ae`, `15ce4f0`). No external registry. | small (writing) / medium (external registry) |
| C06 | P1 | reproducibility: machine-specific paths | in progress | The uncommitted README uses placeholders and documents `NEURO3D_LAB_OUT`. `experiments/run_d2nn_digits.py` line 46 still hard-codes a D: path. | medium |
| C07 | P1 | test and self-test counts | in progress | The uncommitted README gives 32 unit tests and 15 of 15 self-test steps. The committed README still says 29 and 14/14. No versioned test or self-test log exists. | small |
| C08 | P1 | version requirements | in progress | The uncommitted README says Python 3.10 or newer; the committed README says 3.13. CI runs Python 3.13. No run record for other versions. | small |
| C10 | P1 | stated capability | in progress | The uncommitted README describes the add-on as laying out the bench and showing results. The add-on has four operators: build, export, import and run. | small |
| C11 | P1 | scope: D2NN wavelength | in progress | The uncommitted README states lambda = 0.75 mm (terahertz range) and dx/lambda = 0.53. The committed README omits the wavelength. | small |
| C12 | P1 | numerical validity of the D2NN (sampling and periodic edge) | in progress | The uncommitted README discloses the periodic boundary and the sampling limit, and reports T2 and T3. The grid fix is not implemented (see G02, G03). CONTRATO-006c (post hoc, 2026-10-10) shows that the periodic FFT window alone accounts for the V2 annulus discrepancy (0.16 in 12.8–25.6 mm): zero-padded propagation agrees with the Rayleigh–Sommerfeld reference to 1.6 × 10⁻⁹, and input sampling is not a factor. The effect on the D2NN experiments is not yet quantified. | large (CPU, hours); must be done before D2NN results are published |

### 2.4 Additional items found in this review

1. `CONTRIBUTING.md` (committed) says contributions are accepted under Apache-2.0. The Blender add-on is GPL-3.0-or-later. The licence section should name the licence of each directory.
2. `CONTRIBUTING.md` (committed) documents the system temporary folder as the default output location. The working tree now uses `<repo parent>/.cognition/neuro3d-lab` in `blender/neuro3d_lab/bench_io.py`, `experiments/run_validation.py` and `experiments/run_regime_d2nn.py`. The document should follow the code when the change is committed.
3. The negative control (0.49 and 1.57 rad) has no script or output in the repository. `experiments/run_validation.py` contains no sign-reversal branch. Do not cite it as evidence until a script and an output exist.
4. Provenance of CONTRATO-003: `experiments/run_d2nn_digits.py` first appears in commit `af52aff` (2026-10-09 23:11:29 +0200). The run log outside the repository records a start at 23:02:43. The contract should state this order before any TEST result is reported (see G16).
5. The working tree has 25 modified files that are not committed. Statuses marked "in progress" for README items depend on those changes and should be checked again after they are committed.

## 3. Next experiments, in order

Each experiment starts from a pre-registered contract. Thresholds are fixed before any measurement. Failures stay in the record with their values. Exploratory analyses are labelled as exploratory.

1. **CONTRATO-003 TEST evaluation (done: H1 pass, H2 not met, controls pass) and the N = 256 robustness rerun (done, post hoc: stable, TEST 0.9639 to 0.9694 on seed 0, CONTRATO-005).** CONTRATO-007 (fresh writer-disjoint TEST, ten seeds, DEV-selected digital baselines) done: the D2NN reaches 95.1 %, below the CNN at 96.1 %. Still open: a vector-model validation and the larger benchmarks. Gaps: G06, G07, G-21, G02. Start only after the pre-registered DEV grid is complete and the verdict code is frozen. The TEST evaluation runs once, as the contract requires. The N = 256 check retrains the final configuration on a 256 grid (a new training trajectory, see CONTRATO-005) to test sensitivity to padding. It is post hoc, it is labelled as such, and it does not change the pre-registered verdict.
2. **Independent Rayleigh-Sommerfeld reference (G01).** Compare the propagator with direct summation of the Rayleigh-Sommerfeld integral, or with an oversampled Fresnel integral, on the D2NN grid. Use random phase screens and digit inputs. Pre-register the tolerance for each detector region, and log the evanescent energy fraction at each propagation step.
3. **Vector solver on one element (G04).** Benchmark one two-layer element against a vector solver such as FDTD (for example Meep), for TE and TM polarisation, at the same dimensionless parameters. Report the angle at which the scalar error exceeds a pre-registered threshold. Restrict claims to the regime below that angle.
4. **MNIST and Fashion-MNIST under the Lin et al. (2018) protocol (G05, G07).** Run the same protocol on both datasets, with a batched implementation of the forward pass and gradient. Compare with published simulated accuracies only after those values are verified in the source paper.
5. **Parameter-matched digital baselines (G08).** Train a digital multilayer perceptron, a small convolutional network, a linear readout on the same detector regions, and a linear model on the field intensity. Match parameter counts to the D2NN, and use the same split and training budget.
6. **At least 10 seeds with paired statistics (G06, G07, G-21).** Report per-seed Wilson intervals, the mean and sample standard deviation over seeds, and a paired bootstrap or McNemar comparison over all seeds. Apply a Holm correction across the pre-registered hypotheses and controls.
7. **Hardware validation at microwave or THz scale (G09).** Build a scaled device. The scalar model depends only on dimensionless ratios, so a larger wavelength is a valid analogue. Measure detector-region intensities for a fixed set of digit inputs. Report the correlation with simulation, the per-region error and the classification accuracy. Until this step is done, the project makes no hardware claim.

## 4. Collaboration

### 4.1 What external researchers can contribute

- **Reference calculations.** Independent Rayleigh-Sommerfeld or vector-solver results on the same dimensionless parameters (G01, G04).
- **Reimplementation.** An independent forward pass and gradient, for example in PyTorch or JAX, matched to optlab on one configuration within a stated tolerance (G17).
- **Replication.** Clean-clone runs on another machine or operating system, compared with the committed summaries within stated tolerances (G14, G17).
- **Hardware.** Access to microwave or THz equipment for a scaled device (G09).
- **Baselines and datasets.** Parameter-matched digital models, and the MNIST and Fashion-MNIST protocol (G05, G08).
- **Blender add-on.** Testing on Blender 4.5, review of the manifest, and extension of the self-test (G-11, G-12, G-13).
- **Literature.** Related work, and a check of every citation before submission (G18).
- **Language.** English review of the documentation after the translation is complete (G-01).

### 4.2 How to contribute

Read [CONTRIBUTING.md](CONTRIBUTING.md). Changes that affect results, physical models, the `optlab.bench/1` format or the public API start with an issue. Thresholds are not changed after a result is seen. Failures stay in the record. Exploratory work is labelled.

Until `CONTRIBUTING.md` is corrected (section 2.4, item 1), the licence files apply by directory: the root [LICENSE](LICENSE) (Apache-2.0) covers the core, and [blender/neuro3d_lab/LICENSE](blender/neuro3d_lab/LICENSE) (GPL-3.0-or-later) covers the Blender add-on.

## 5. Out of scope

- **Physical results and device claims.** Out of scope until the hardware step (section 3, item 7) is complete. No measured data are used in this project.
- **Speed, energy or throughput comparisons** with digital or other optical systems. No timing or energy figures have been measured.
- **Vector and polarisation effects** in the current model. They are listed as limitations in `docs/MODEL.md` section 8 and are addressed only through G04.
- **Reflections between planes, finite bandwidth and spectral dispersion.** These are listed as limitations in `docs/MODEL.md` section 8.
- **Fabrication effects:** quantised phase levels, alignment and fabrication tolerance. These are P2 items in the audit and are not scheduled.
- **Training or optimisation inside Blender.** D2NN training runs in Python (`optlab.d2nn`). Blender displays results and casts rays for path checks only.
- **Computing the complex field inside Blender.** Only optlab computes it.
- **Claims of absolute accuracy, novelty or priority.** Comparisons with published numbers are made only after those numbers are verified in their source.
- **The TEST accuracy of CONTRATO-003.** It is not reported in this roadmap.
