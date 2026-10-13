# CONTRATO-008 (pre-registered: MNIST and Fashion-MNIST)

**Status:** pre-registered. Written 2026-10-10 **before** any run of `experiments/run_d2nn_mnist_torch.py` or `experiments/run_mnist_baselines.py`. No script has been run, and no model has read a TEST image or label. The commit that adds this file is the pre-registration commit; its hash will be recorded under Results when it is made.

**Question.** Does the three-layer phase-only D2NN of CONTRATO-003, with the same optics, classify MNIST and Fashion-MNIST (official 10,000-image test sets) better than four digital baselines trained on the same splits with the same budget?

**Scope of the claim.** Scalar simulation only. Every D2NN number carries the periodic-window caveat of CONTRATO-006c: the FFT propagator wraps around a 51.2 mm window at dx = 0.4 mm. CONTRATO-010 (pending) will address it. The N = 128 grid effect noted in CONTRATO-003 (note 3) and the N = 256 robustness check of CONTRATO-005 also apply. No D2NN accuracy from this contract is a publishable claim without these caveats.

## Data

Official IDX files, gzip-compressed, in `D:\PROJECTS\.cognition\neuro3d-lab\datasets\mnist\` and `D:\PROJECTS\.cognition\neuro3d-lab\datasets\fashion\`. Fashion-MNIST comes from the zalandoresearch/fashion-mnist GitHub repository. Each folder holds the four files below.

SHA-256 values were computed with `sha256sum` on the local files. The scripts check SHA-256 before reading each file and stop on a mismatch. MD5 was also computed locally. The MNIST MD5 values match the torchvision values. No download was made in this contract, so the Fashion-MNIST MD5 values are recorded as computed and were not checked against upstream.

| Dataset | File | Bytes | SHA-256 | MD5 |
|---|---|---|---|---|
| MNIST | train-images-idx3-ubyte.gz | 9,912,422 | 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609 | f68b3c2dcbeaaa9fbdd348bbdeb94873 |
| MNIST | train-labels-idx1-ubyte.gz | 28,881 | 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c | d53e105ee54ea40749a09fcbcd1e9432 |
| MNIST | t10k-images-idx3-ubyte.gz | 1,648,877 | 8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6 | 9fb629c4189551a2d022fa330f9573f3 |
| MNIST | t10k-labels-idx1-ubyte.gz | 4,542 | f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6 | ec29112dd5afa0611ce80d1b7f02629c |
| Fashion | train-images-idx3-ubyte.gz | 26,421,880 | 3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84 | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| Fashion | train-labels-idx1-ubyte.gz | 29,515 | a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845 | 25c81989df183df01b3e8a0aad5dffbe |
| Fashion | t10k-images-idx3-ubyte.gz | 4,422,102 | 346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073 | bef4ecab320f06d8554ea6380940ec79 |
| Fashion | t10k-labels-idx1-ubyte.gz | 5,148 | 67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5 | bb300cfdad3c16e7a12a480ee83cd310 |

## Splits

- **TRAIN:** official training images 0 to 49,999 (50,000), in file order.
- **DEV:** official training images 50,000 to 59,999 (10,000).
- **TEST:** the 10,000 official test images. Evaluated once per model and seed, after all selection is frozen (see TEST rule).
- **Selection subset:** the first 10,000 images of TRAIN.

## Input encoding (as CONTRATO-003, with a 28 → 64 resize)

1. Pixel values divided by 255, giving [0, 1] (float64).
2. Bilinear resize 28 × 28 → 64 × 64 with `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)`, then clamp to [0, 1].
3. Normalise to unit L2 energy over the 64 × 64 image: Σ|u|² = 1.
4. D2NN only: zero-pad 32 pixels per side to 128 × 128. Complex amplitude, initial phase 0.

The baselines use the output of step 3 only (64 × 64, no padding).

## D2NN architecture (identical to CONTRATO-003)

- N = 128, DX = 0.4e-3 m, LAM = 0.75e-3 m, DIST = [18e-3] × 4. This gives four propagations and L = 3 phase layers, with 3 × 128 × 128 = 49,152 phase parameters.
- Phase layers: t = exp(iφ). Initial phases come from `np.random.default_rng(seed).uniform(0, 2π, size=(3, 128, 128))`, the same call as the optlab constructor (`src/optlab/d2nn.py`, line 36).
- Propagation: FFT-based, with transfer function H = exp(i d kz) for propagating components and zero for evanescent ones. The definition is that of `experiments/replicate_d2nn_pytorch.py` (CONTRATO-009). It is copied, not imported, because that script sets its constants at module level.
- Detector: `build_masks()` is imported from `experiments/run_d2nn_digits.py` and not re-derived. Ten 12 × 12 pixel windows (4.8 mm), at columns −40, −20, 0, 20, 40 px and rows +20, −20 px. Class k goes to column k mod 5 and row k div 5.
- Scores: s_c = β · Σ mask_c · I, with I = |f|². Loss: cross-entropy on s.

Implementation (not part of the physics): batched torch float64 on CPU, not optlab's per-sample code. The intensity is computed as Re(f)² + Im(f)², which equals |f|². The loss is the mean over the mini-batch.

## Training

- Adam with β1 = 0.9, β2 = 0.999, ε = 1e-8, one step per mini-batch of 32. The last mini-batch of an epoch is not dropped (TRAIN of 50,000 leaves 16 images in the last batch).
- Shuffling: one generator per run, `np.random.default_rng(seed + 1000)`, drawing one permutation per epoch. For a given seed, the D2NN and the baselines see the same batch order.
- The final-epoch model is used. No early stopping.
- Baselines: `torch.manual_seed(seed)` before construction. Adam with lr = 1e-3, the same betas and ε, batch 32, 20 epochs, float64.

## Baselines

| Name | Architecture | Parameters |
|---|---|---|
| logreg | Multinomial logistic regression: Linear 4096 → 10 | 40,970 |
| mlp12 | Linear 4096 → 12, ReLU, Linear 12 → 10 | 49,294 |
| mlp128 | Linear 4096 → 128, ReLU, Linear 128 → 10 | 525,706 |
| cnn | Conv 1 → 8 (3 × 3, pad 1), ReLU, MaxPool 2; Conv 8 → 16 (3 × 3, pad 1), ReLU, MaxPool 2; Linear 4096 → 10 | 42,218 |

mlp12 is parameter-matched to the D2NN (49,294 against 49,152). mlp128 is a stronger digital reference with about 10.7 times the D2NN parameter count. It is not parameter-matched.

## Selection (DEV only, seed 0, per dataset)

- Training on the first 10,000 TRAIN images for 5 epochs. The score is the DEV accuracy after epoch 5.
- Grid: β ∈ {20, 50, 100} × lr ∈ {0.02, 0.05}. Every configuration starts from the same seed-0 phases.
- Winner: highest DEV accuracy. Ties go to the smaller β, then the smaller lr.
- The selection file is frozen: the `select` stage refuses to overwrite it.
- Edge rule. Every edge of both grids counts: β ∈ {20, 100} (smallest or largest) and lr ∈ {0.02, 0.05} (smallest or largest). The `select` stage records in its output whether the chosen β and lr are on an edge (field `selected.edge`: `beta`, `lr`, `any`). If the choice is on an edge, the grid is not extended and the edge choice is reported as a limitation, not corrected. This follows CONTRATO-003, note 2.

## Final runs (per dataset, seeds 0, 1, 2)

- Full TRAIN (50,000), 20 epochs, with the selected β and lr. Phase initialisation and shuffling follow the seed rules above.
- DEV accuracy is logged at the end of each epoch.
- TEST is evaluated once, after training, under the TEST rule.

## Baseline runs (per dataset, seeds 0, 1, 2)

- The four models above, on the same splits, for 20 epochs. DEV is logged per epoch. TEST is evaluated once per model under the TEST rule.
- No DEV selection: the hyperparameters are fixed in this contract.

## Control C1 (per dataset)

- D2NN seed 0, with the selected β and lr, full TRAIN, 20 epochs, and the TRAIN labels permuted by `np.random.default_rng(100 + 0).permutation(labels)`.
- Evaluated on DEV only. TEST is not used.
- Pre-registered expectation: DEV accuracy within 0.03 of 0.10, that is 0.07 to 0.13. This is an expectation, not a verdict.

## TEST rule (one use per model)

- Only the `final` and `baselines` stages read TEST, and only after the claim described here.
- **Marker.** `D:\PROJECTS\.cognition\neuro3d-lab\results\test_done_<dataset>_<model>_seed<k>.marker`, one file per dataset, model and seed. It is created exclusively (`os.open` with `O_CREAT | O_EXCL | O_WRONLY`) before any TEST prediction, so runs of different models do not share a file and cannot race. The model names are `d2nn`, `logreg`, `mlp12`, `mlp128` and `cnn`.
- **Refusal.** If creation fails with `FileExistsError`, TEST is refused and the script exits with a message. Before any training, both stages also refuse to start if a marker for that dataset, model and seed already exists. The `final` stage also refuses to start if the phases file exists, and writes it only if it does not exist (open mode `x`); saved phases are never overwritten.
- **Clarification.** The registration says "refuse to evaluate TEST if the marker exists". Read literally, the D2NN (`final` stage) and the baselines (`baselines` stage) of the same seed could never both be evaluated, because they are separate stages. The per-model marker keeps the one-use rule for each model. This clarification was made before any run. No threshold or hypothesis changed.
- If a stage stops after the claim, the model counts as used.
- The `select` and `c1` stages never read TEST.

## Statistics

- **Wilson 95 % interval** (z = 1.96, `optlab.stats.wilson_interval`) for the TEST accuracy of every model, dataset and seed (n = 10,000).
- **Across seeds:** mean and sample standard deviation of TEST accuracy. No binomial interval is reported for the seed mean, because the seeds share one TEST set. Such an interval would be mis-specified (CONTRATO-003, note 4).
- **McNemar, exact, paired, TEST seed 0 only:** D2NN against each of the four baselines. b = D2NN correct and baseline wrong. c = D2NN wrong and baseline correct. p is the two-sided binomial test of min(b, c) out of b + c with probability 0.5, and p = 1 when b + c = 0.
- **Holm** step-down correction across the four comparisons of each dataset. The two datasets are separate families.

## Pre-registered verdict rule

For each dataset and each baseline, the D2NN is claimed to beat that baseline only if both of the following hold:

1. the Holm-adjusted McNemar p is below 0.05, and
2. the D2NN is more accurate than that baseline on TEST seed 0.

Otherwise the verdict is "does not beat". A non-significant result is not evidence of equal accuracy. Seeds 1 and 2 are reported without a test. The verdict covers only the four baselines listed here. A stronger CNN or MLP is not tested.

**Caveat, applies to every D2NN number in this contract:** the periodic-window caveat (CONTRATO-006c). CONTRATO-010 (pending) will address it.

## Scope and consequences

- Scalar simulation only. Fabrication noise, misalignment, inter-plane reflections, spectral dispersion and hardware are not covered.
- This contract does not check that the batched torch code matches optlab at N = 128. CONTRATO-009 checked the same propagator and gradient at N = 32. Whether to run an equivalence check before the `select` stage is a decision for the project owner. It is not part of this protocol.
- Clarifications made before any run (no threshold, hypothesis or grid changed):
  - (a) the per-model TEST markers above;
  - (b) the baselines use float64, because the protocol fixes float64 only for the D2NN;
  - (c) the selection file, the saved phases and each TEST claim are protected against overwriting;
  - (d) the encoding is computed per mini-batch, not stored in memory, to keep the memory footprint small.

## Commands

```
python -B experiments/run_d2nn_mnist_torch.py --dataset {mnist,fashion} --stage select --seed 0
python -B experiments/run_d2nn_mnist_torch.py --dataset {mnist,fashion} --stage final  --seed {0,1,2}
python -B experiments/run_d2nn_mnist_torch.py --dataset {mnist,fashion} --stage c1     --seed 0
python -B experiments/run_mnist_baselines.py  --dataset {mnist,fashion} --stage baselines --seed {0,1,2}
```

`select` runs first. `final`, `c1` and `baselines` follow, in any order.

## Outputs

- **Repository:** `experiments/results/mnist_fashion_summary.json`. Every stage rebuilds it from the per-run files.
- **Local, not versioned:** `D:\PROJECTS\.cognition\neuro3d-lab\results\`
  - `<dataset>_select.json` and `<dataset>_select_seed0.log`;
  - `<dataset>_final_seed<k>.json`, `<dataset>_final_seed<k>_phases.npy` and `.log`;
  - `<dataset>_baselines_seed<k>.json` and `.log`;
  - `<dataset>_c1_seed0.json` and `.log`;
  - `test_done_<dataset>_<model>_seed<k>.marker`, the TEST markers.
- Per-run files hold the DEV history, the TEST predictions and per-sample correctness (for McNemar), parameter counts and timings.

## Results

None yet. No script has been run. Pre-registration commit: to be recorded.

## Gate T0 (added before any run of this contract)

Before any training, the batched torch propagation must match optlab at N = 128. Script `experiments/check_torch_vs_optlab_n128.py`, output `experiments/results/check_torch_vs_optlab_n128.json`. It uses the thresholds of CONTRATO-009: intensities 1e-9, scores 1e-9, loss 1e-10, gradient 1e-8 (relative or absolute as there). The run on 2026-10-10 gave intensities 4.3e-14, scores 4.2e-15, loss 3.1e-15 and gradient 5.8e-14: **PASS**. The MNIST and Fashion runs refuse to start unless this gate has passed for the current code. No TEST data is used by the gate.

## Results archived on 2026-10-13 (run of 2026-10-10, interrupted and incomplete)

The run was made on 2026-10-10 under this contract and was not committed at the time. Its local outputs are archived unchanged in `experiments/results/contrato-008/` (checksums in `SHA256SUMS.txt`) and the summary written by the run is `experiments/results/mnist_fashion_summary.json`. Nothing was re-run or edited for this archive.

**Completeness.** Selection, D2NN final seeds 0–2 and baseline seed 0 finished for both datasets. Baseline seeds 1–2 were not run. The permuted-label control C1 was interrupted at epoch 6 of 20 on both datasets (`*_c1.log`) and has no result. The contract is therefore **incomplete**; H1, H2 and C1 cannot be decided as registered. It is not completed now because its periodic-window model is superseded (CONTRATO-010, CONTRATO-011, CONTRATO-013); the decision was put to JEV (provenance `jev`, 0.49, indifferent) and the archive was preferred to new runs under a superseded model.

| Dataset | D2NN TEST (seeds 0, 1, 2) | D2NN mean (sample SD) | logreg s0 | mlp12 s0 | mlp128 s0 | cnn s0 |
|---|---|---|---|---|---|---|
| MNIST | 0.9321, 0.9332, 0.9348 | 0.9334 (0.0014) | 0.9276 | 0.9341 | 0.9766 | 0.9791 |
| Fashion-MNIST | 0.8188, 0.8274, 0.8291 | 0.8251 (0.0055) | 0.8387 | 0.8413 | 0.8761 | 0.8916 |

Seed-0 exact McNemar with Holm over four baselines (from the summary): on MNIST the D2NN beats logreg (Holm p = 0.047) and does not beat mlp12, mlp128 or cnn; on Fashion-MNIST it beats none. Selected D2NN setting β = 100, lr = 0.02 on both datasets, on the edge of both grids (limitation recorded by the run).

**Consequences.** (1) All D2NN numbers above use the periodic window that CONTRATO-010/011 found material; they are historical. (2) The official MNIST and Fashion-MNIST TEST10K files were examined by this run, so later protocols must treat them as re-used tests (CONTRATO-013).
