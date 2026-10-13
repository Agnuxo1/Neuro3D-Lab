# CONTRATO-013 — Closure benchmark under the finite-support scalar model (pre-registered)

**Status:** pre-registered on 2026-10-13, **before** any DEV selection, final training or TEST evaluation under this protocol. The commit that adds this file, together with the configurations, tools and evidence it names, and its push to `origin`, are the registration; the hash is recorded under *Registration*. No model has been trained with these configurations. Thresholds are never changed after results are seen. An independent reviewer (separate agent, no part in the drafting) checked a first draft against the code and the evidence; every point raised is addressed in this version.

**Relation to earlier contracts.** CONTRATO-003 to CONTRATO-011 are kept unchanged as historical records. They use the periodic 51.2 mm FFT window that CONTRATO-010 and CONTRATO-011 found material. This contract does not reinterpret them. The interrupted CONTRATO-008 run (2026-10-10) is archived in `experiments/results/contrato-008/`; its TEST use matters here (section 3).

## 1. Questions, hypotheses and verdict rules

- **H1 (absolute accuracy, official TEST10K).** Mean D2NN accuracy ≥ 0.90 on MNIST with every seed ≥ 0.88; mean ≥ 0.75 on Fashion-MNIST with every seed ≥ 0.70. These thresholds are the explicit proposals of the completion guide (`completion-configs/closure_*.json`), not literature standards. **Disclosure:** when they were registered, the unpublished CONTRATO-008 run (periodic model, different encoding) had produced D2NN TEST means of 0.933 (MNIST) and 0.825 (Fashion), known to the registering agent.
- **H2 (comparison).** The D2NN beats the DEV-selected primary digital baseline: higher mean accuracy, positive seed-0 difference, Holm-adjusted exact McNemar p < 0.05 over the five digital comparisons of the dataset, and a positive lower end of the 95 % crossed bootstrap interval (2,000 draws over paired seeds and examples). "Not significant" is not evidence of equivalence.
  - **Fashion-MNIST:** H2 is decided on TEST10K by `benchmark summarize` (example-level bootstrap; no author identifiers exist).
  - **MNIST:** H2 holds only if it holds on TEST10K (`benchmark summarize`) **and** on QMNIST50K with the bootstrap grouped by NIST writer (`scripts/completion_tools/q50k_verdicts.py`: Holm over the five seed-0 McNemar p-values on Q50K, writer-grouped crossed bootstrap, same primary baseline). If the two disagree, MNIST H2 is reported as **not supported** and both results are shown.
- **C1–C3 (controls, TEST10K).** Permuted-label D2NN mean ≤ 0.15; untrained random phases mean ≤ 0.20; propagation without phase layers mean ≤ 0.20. `no_layers` has no trainable parameter, so its ten seeds are identical by construction.
- **Secondary (descriptive, no verdict):** H1-type means on Q50K by HSF series (series 1 is absent from MNIST TRAIN and TEST and probes distribution shift); writer-grouped TEST10K comparisons (`q50k_verdicts.py` on `test.npz`).
- **W1 (window check, DEV only).** For the seed-0 D2NN of each dataset, evaluating the frozen phases at padding 8 instead of 4 (refinement 1) changes at most 0.5 % of DEV labels and at most 1e-2 of the scores (relative L2) (`fidelity_study.py evaluate` + `window_check.py`). If W1 fails, every D2NN result of that dataset is labelled "window-unconverged".
- **Refinement line (DEV only, descriptive).** The same seed-0 phases evaluated at refinement 2, 4 and 8 (padding 2): DEV accuracy and label changes. This links the benchmark masks to CONTRATO-014.

## 2. Physical model and scope

| Quantity | Value |
|---|---|
| Physical support | n = 128 pixels per side, pixel 0.4 mm (0.533 λ) |
| Wavelength | λ = 0.75 mm (about 0.40 THz) |
| Geometry | four propagations of 18 mm, three phase-only layers, 3 × 128² = 49,152 phases |
| Exterior | `aperture`: light outside the 51.2 mm support is absorbed at every plane |
| FFT window | padding p = 4 (204.8 mm window, m = 512) |
| Quadrature | refinement r = 1 (one numerical sample per physical pixel) |
| Spectral filter | none; evanescent components removed |
| Arithmetic | float64 / complex128 |

**Scope of every number in this contract: the sampled scalar model** (one sample per 0.533 λ pixel). It is not a converged simulation of a fabricated piecewise-constant element: with the historical masks, refinement 1 → 2 changes detector scores by 270 % and 2 → 4 by 28 % (CONTRATO-012). The consequence for trained masks is the subject of CONTRATO-014 and of the refinement line above. No statement about a physical device, its speed or its energy follows from this contract.

**Window adequacy is established for decisions, not for fields.** With the historical masks and 64 writer-disjoint DEV images (`evidence/boundary-historical-dev64-r1.json`), padding 4 → 8 changes detector scores by 0.15 % and no label, but the field by 2.6 % and the intensity by 2.1 %; padding 8 → 16 changes them by 0.6 % and 0.5 %. The guide's window criteria for fields (≤ 1e-2) and intensities (≤ 2e-2) therefore **fail at p = 4** and pass at p = 8, while the score (≤ 1e-2) and label (≤ 0.005) criteria pass at p = 4. This contract makes claims only about scores and labels; W1 re-checks this on the trained masks.

**Why refinement 1 and not the guide proposal r = 4.** The registered configurations are identical to the guide files except `study_id` and `refinement` (1 instead of 4); the guide already proposed padding 4. Measured on the RTX 3090 (`evidence/gpu-probe-rtx3090.json`): 0.0316 s per training sample at m = 2048, about 250 GPU-hours per dataset for this design, and refinement 4 is still not converged (r 4 → 8 changes scores 6 %). At m = 512: 0.0020 s per sample, about 16 GPU-hours per dataset. The choice was put to JEV with these facts (`evidence/jev-protocol-decision-20261013.json`, provenance `jev`, option A, confidence 0.85). The guide files `closure_mnist.json`, `closure_fashion.json` and `closure_mnist_writer.json` are kept unchanged and are **not** run.

## 3. Data

| Dataset | TRAIN | DEV | TEST | Writers |
|---|---|---|---|---|
| MNIST, writer-disjoint (`data/mnist-writer`, split from QMNIST metadata) | 49,905 (448 writers) | 10,095 (91 writers), 0 shared with TRAIN | official 10,000 (297 writers, disjoint from TRAIN and DEV) | NIST writer and HSF series per image |
| QMNIST50K (`data/mnist-writer/q50k.npz`) | — | — | 50,000 (488 writers, 0 shared with TRAIN/DEV, 250 shared with TEST10K; series 0/1/4 = 13,450/11,956/24,594) | yes |
| Fashion-MNIST (`data/fashion`) | official train 0–49,999 | official train 50,000–59,999 | official 10,000 | not available |

Source files were admitted by SHA-256 against the values in `neuro3d_completion/data.py` and `neuro3d_completion/qmnist.py`; all matched (`data/*/manifest.json`). Registered split hashes:

| Split | SHA-256 |
|---|---|
| mnist-writer/train.npz | `31735602d26563a24af43dc8e94d0ac09bcc28b9c7285ff699292c8aa14e7c37` |
| mnist-writer/dev.npz | `d50edbdaa9b85563a3062b5cabac95b8a18369e226434867b3b6cf4b1152ac09` |
| mnist-writer/test.npz | `051a2ec47559a2b7cb3356e14e260b6579e29028604827f2592f4e7392dd3038` |
| mnist-writer/q50k.npz | `fb1ff5b2e522655135af7a14744d7c21e6f5fb44b99889fc5ece4dc2fce624f4` |
| fashion/train.npz | `31aae3fdce6bec06cc5b03dd662040774eb018f3e43f013bab2632978b0e2e51` |
| fashion/dev.npz | `9026137a3a9e0501565389f9a5007a571c462712634e66d9044d449cd74c7be5` |
| fashion/test.npz | `4241d7abbfceb8108637640067e1ed85b151f6ed6e69553dd60f1157fc97f531` |

- **Contiguous-DEV MNIST protocol not run.** `closure_mnist.json` is superseded by the writer-disjoint split, which reproduces how the official TEST is built. Decided before any selection.
- **Input encoding.** `data.encode` (linear `scipy.ndimage.zoom` 28 → 64, unit energy), identical for every model. It differs from the CONTRATO-008 encoding (median per-image L2 difference 8.9 %, `evidence/v2-review/encoding.json`).

### TEST status (declared before any use)

- The official MNIST and Fashion-MNIST TEST10K files were evaluated on 2026-10-10 by the CONTRATO-008 run (periodic model; D2NN seeds 0–2, baselines seed 0). **They are re-used, not fresh**, and every result on them is labelled "re-used TEST".
- **No model of this project has been evaluated on QMNIST50K images.** Its QMNIST metadata (writer and series identifiers of the training file) was used to design the writer-disjoint split. It is evaluated once, after all 90 MNIST final runs exist, by `qmnist evaluate`, which reserves every use in its own ledger before reading the images; `verify_frozen.py --q50k` must first confirm its registered SHA-256.
- Exploratory tools (`guards.refuse_reserved_split`) refuse any file whose name or recorded hash is a TEST or Q50K split.

## 4. Models

| Name | Definition | Trainable parameters |
|---|---|---|
| d2nn | the model of section 2 | 49,152 |
| linear | Linear 4096 → 10 | 40,970 |
| mlp12 | 4096 → 12 → 10, ReLU | 49,294 |
| mlp128 | 4096 → 128 → 10, ReLU | 525,706 |
| cnn | Conv(1→8,3) ReLU Pool, Conv(8→16,3) ReLU Pool, Linear 4096 → 10 | 42,218 |
| quadratic12 | 12 learned features squared, positive readout | 49,272 |
| permuted | d2nn trained on permuted TRAIN labels | 49,152 |
| untrained | d2nn with random initial phases, no training | 49,152 |
| no_layers | one 72 mm propagation, no phase layer | 0 |

## 5. Procedure

1. **Selection (DEV, seed 0).** Full TRAIN, 20 epochs, batch 16, Adam, last epoch. D2NN grid β ∈ {100, 200, 400} × lr ∈ {0.02, 0.05}; digital grid lr ∈ {0.001, 0.003}. Winner: highest last-epoch DEV accuracy; ties go to the earlier grid point. The primary baseline for H2 is the digital model with the highest DEV accuracy; ties go to the first in the order linear, mlp12, mlp128, cnn, quadratic12. **Known limitation:** CONTRATO-008 selected the smallest β and lr of its grid; the grid here is not extended after selection, a winner on an edge is reported as a limitation, and the D2NN gets six grid points against two per digital model. Run with `scripts/completion_tools/select_sharded.py` (one process per grid point, same `benchmark.train`; identity with `benchmark.select` tested in `tests/test_completion_tools.py`), every shard on CUDA in deterministic mode.
2. **Freezing.** The assembled record (`evidence/mnist-writer-selected.json`, `evidence/fashion-selected.json`) stores protocol, settings, all DEV histories, train/dev/test hashes, the NumPy/Torch equivalence gate and the package code digest. `benchmark final` refuses to run if code or data change afterwards.
3. **Final runs.** Seeds 0–9 for every model and control: `python -m neuro3d_completion.benchmark final --frozen <frozen> --output-dir <dir>` with the **pinned output directories** `completion_results/mnist-writer` and `completion_results/fashion` (the TEST ledger lives in `<dir>/test_ledger`; no other directory is used). Optical models on CUDA with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set by `queue_driver.py`); digital models on CPU (selected on CUDA; both float64 and deterministic); the device is recorded per run. Seed 0 of each final run repeats the selection run of its winner, so its TEST result is not independent of the choice of settings.
4. **Failure policy.** A run that fails before its TEST reservation is repeated after its partial directory is moved (not deleted) to `completion_results/_failed/`. A failure after the reservation is final: the run is recorded as failed with its log, never re-evaluated, and the summaries of that dataset are produced over the remaining seeds with an explicit amendment note.
5. **Summaries.** Before each step, `scripts/completion_tools/verify_frozen.py` checks that the package code and the splits are those frozen by the selection (record kept in `evidence/`). Then `benchmark summarize` per dataset; for MNIST, `q50k_verdicts.py` on TEST10K (writer-grouped, secondary).
6. **Fresh test.** `verify_frozen.py --q50k`, then `qmnist evaluate` on Q50K once, then `q50k_verdicts.py` on Q50K (H2-Q50K) and `qmnist summarize`.
7. **Secondary, exploratory, DEV only** (not confirmatory): W1 and the refinement line; robustness of the seed-0 D2NN masks (`robustness.py`); detection-noise curves (`photon_budget.py`); per-pixel influence (`expressivity.py`). Any further analysis of TEST or Q50K predictions beyond the registered tools is done blind (`blind.py`).

## 6. Statistics and power

- Per seed: accuracy, Wilson 95 % interval (valid for one model on one set of independent images). Across seeds: mean, sample SD, minimum. No binomial interval on seed means.
- Seed 0: exact two-sided McNemar for the D2NN against each of the five digital models; Holm over those five.
- Crossed bootstrap (2,000 draws) over paired seeds and examples; on MNIST TEST10K (secondary) and Q50K (H2-Q50K) grouped by writer.
- **Design power** (`evidence/power-design-n10k.json`, `-n50k.json`; exact McNemar, power 0.8, α = 0.05): minimum detectable accuracy difference at n = 10,000 is 0.40–0.90 points for discordance 0.02–0.10 (0.49–1.09 at the worst-case Holm level); at n = 50,000, 0.18–0.40 points. These assume independent images. On the writer-disjoint DEV the measured design effect of an error rate is 4.4–5.8 (effective n ≈ 1,700–2,300 per 10,095; `evidence/v2-review/writers.json`), so for inference to new writers the detectable differences are larger.

## 7. Compute plan

About 16 GPU-hours per dataset for the optical runs at 0.002 s per sample, run through the workstation's FIFO queue as separate tickets (one grid point or one final run per ticket) by `queue_driver.py` with the job lists of `make_jobs.py`. Nothing is skipped silently: the driver stops at the first failure.

## 8. Registration

- Pre-registration commit: `9521a9bfd30a400479015ff3b70ba9454a314131` (2026-10-13), pushed to `origin/completion/2026-10-12` before any run of this contract; hash manifests in `evidence/protocol-manifest-*.json` (commit b7f8202).
- External registry (OSF or Zenodo): pending; it needs the owner's account.

## 9. Results

To be filled after the runs, without changing sections 1–7.
