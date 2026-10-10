# CONTRACT-007: fresh-test benchmark of the D2NN on the UCI optdigits training file

## Status and provenance

- **Pre-registration:** written 2026-10-10 (UTC), before any full run. A dry run has been executed (no training, no results). The full run has **not** been executed.
- **Commit:** this document and `experiments/run_fresh_test_benchmark.py` must be committed before `--run` is executed. The commit hash is recorded in the result registry at the end of this file, as in CONTRACT-003.
- **Relation to earlier contracts:** CONTRACT-003 used the 1797-sample `load_digits` source for TRAIN, DEV and a 360-sample TEST, and its TEST was spent. CONTRACT-004 measured the grid sensitivity of the N = 128 window. CONTRACT-005 (post hoc) found the seed-0 TEST accuracy stable at N = 256. This contract uses a source that CONTRACT-003 never touched.
- **DEV is not fresh.** CONTRACT-003 used DEV to choose β and the learning rate. Its DEV values at lr = 0.02 were β = 20: 0.9081, β = 50: 0.9443, β = 100: 0.9554 (`experiments/results/d2nn_digits_summary.json`). Those values justify extending the grid upward (β ∈ {100, 200, 400}) and dropping β = 20 and 50, which were already lower on DEV. Only TEST_FRESH is fresh.
- **Decision record (JEV).** Four unfixed baseline details were put to the JEV router (`router.py jev`, `provenance=jev`, `status=connected`, model `jev-1.13.0`, `remote_decision=true`). Three of its four answers had confidence below 0.5. Two of its answers conflict with the protocol fixed before the consult, so they were not adopted. The adopted choices and the reasons are below.

| Question | JEV answer (confidence) | Adopted | Reason |
|---|---|---|---|
| CNN selection | Fixed final epoch 60, no DEV selection (0.73) | Epoch with the highest DEV accuracy among the 60, seed 0, ties to the earlier epoch | The protocol fixed before the consult says "selected by DEV". The JEV answer is recorded as a divergence. |
| MLP selection | Single alpha 1e-4, no selection (0.41) | L2 alpha in {1e-4, 1e-3, 1e-2}, selected on DEV with seed 0 | Same reason. |
| Tie-break for the best baseline | Earlier in the order (a) to (d) (0.19) | More trainable parameters | Low JEV confidence. This rule makes H2 harder to pass, so a tie cannot favour the D2NN. |
| Seeds for control C2 | Seeds 0 to 2, verdict on the mean (0.16) | Seeds 0 to 9, verdict on the mean, every seed reported | Matches the 10 D2NN seeds. Forward passes only, so the cost is small. |

## Question

Does a three-layer phase-only D2NN, configured and trained as in CONTRACT-003, classify digits from samples that no earlier analysis used, reaching the accuracy threshold of H1 over ten seeds, and does it do at least as well as the best of four baselines trained on the same TRAIN split (H2)?

## 1. Data

### 1.1 Sources and the writer question

- **Source (1797 samples).** `sklearn.datasets.load_digits`. These samples are the UCI optdigits test file (verified before this contract was written). The script checks it again at run time: every one of the 1797 samples (pixels and label) must match exactly one row of OpenML `optdigits` version 1 (5620 rows). The dry run confirmed 1797 of 1797 exact matches.
- **TEST_FRESH (3823 samples).** The OpenML rows that did not match the source, which are the UCI optdigits training file (3823 samples). The script asserts the count.
- **Writer sets, UCI documentation** (https://archive.ics.uci.edu/dataset/80, read 2026-10-10). The page says: "From a total of 43 people, 30 contributed to the training set and different 13 to the test set." The word "different" implies that the two writer sets are disjoint, but the page does not state the disjointness explicitly.
- **Writer effect: not verified at sample level.** The data carry no writer identifiers, so the disjointness is taken from the documentation and is not checked against the samples. If the documentation holds, TEST_FRESH comes from 30 writers who are not among the 13 writers of the source, and therefore from writers not represented in TRAIN or DEV.

### 1.2 Splits

| Split | Size | Origin | Use |
|---|---|---|---|
| TRAIN | 1078 | CONTRACT-003 split of the source, stratified, `random_state` 2026 | Training of every model |
| DEV | 359 | CONTRACT-003 split of the source, same procedure | Selection of every hyperparameter |
| TEST (CONTRACT-003) | 360 | CONTRACT-003 split of the source | Not evaluated here. Counted only. |
| TEST_FRESH | 3823 | UCI optdigits training file (unmatched OpenML rows) | Evaluated once per configuration and seed, after all selection |

The split code is copied from `experiments/run_d2nn_digits.py`, not imported from it.

### 1.3 Input encodings

- **D2NN:** as CONTRACT-003. Amplitude a = img/16, repeated 8 × 8 with `np.kron` to 64 × 64, normalised to unit L2 norm (u0 = a/‖a‖₂), then zero-padded to 128 × 128.
- **Baseline (a):** the flattened 64 × 64 unit-energy vector (4096 features).
- **Baselines (b), (c), (d):** the raw amplitude a = img/16 of the 8 × 8 image, with no energy normalisation (64 features; the CNN takes 1 × 8 × 8).

## 2. D2NN (fixed)

- **Architecture, as CONTRACT-003:** N = 128, dx = 0.4 mm, λ = 0.75 mm (dx/λ = 0.53), 18 mm between planes, three phase-only layers (t = exp(iφ)) with phases drawn uniformly in [0, 2π) from the run seed. Phase parameters: 3 · 128² = 49,152.
- **Detector:** ten 12 × 12 pixel regions (4.8 mm). Columns at x = −40, −20, 0, 20, 40 pixels and rows at y = +20, −20 pixels. Class k goes to column k mod 5 and row k div 5. Logits are s_k = β · E_k, where E_k is the energy in region k.
- **Loss and gradient:** softmax cross-entropy with the analytic gradient of `optlab.d2nn.DiffractiveNetwork.loss_and_grad`.
- **Training:** Adam (β1 0.9, β2 0.999, ε 1e-8), mini-batches of 32 with the mean gradient per batch, 30 epochs, learning rate 0.02 fixed.
- **DEV grid (seed 0 only):** β ∈ {100, 200, 400} at lr = 0.02. Winner = highest DEV accuracy. Ties go to the smaller β.
- **Final runs:** the winning β, seeds 0 to 9. Each seed trains on TRAIN and is evaluated on DEV and on TEST_FRESH.

## 3. Baselines (fixed grids, selection on DEV only)

Every baseline is trained on TRAIN, selected on DEV, and evaluated once on TEST_FRESH.

| Code | Model | Input | Trainable parameters | Selection on DEV | Ties |
|---|---|---|---|---|---|
| (a) | Multinomial logistic regression (scikit-learn, lbfgs, max_iter 3000) | 64 × 64 unit-energy (4096) | 40,970 | C ∈ {0.01, 0.1, 1, 10, 100, 1000} | Smaller C |
| (b) | Multinomial logistic regression, same settings | 8 × 8 raw (64) | 650 | Same C grid | Smaller C |
| (c) | MLP (scikit-learn): one hidden layer of **653** ReLU units, Adam with learning rate 1e-3, batch 32, max_iter 200, random_state 0 | 8 × 8 raw (64) | **48,985** = 64·653 + 653 + 653·10 + 10 | L2 alpha ∈ {1e-4, 1e-3, 1e-2}, seed 0 | Smaller alpha |
| (d) | CNN (PyTorch, CPU): Conv(1→32, 3 × 3, pad 1), ReLU, MaxPool 2, Conv(32→112, 3 × 3, pad 1), ReLU, flatten to 1792, Linear(1792→10) | 1 × 8 × 8 raw | **50,618** (320 + 32,368 + 17,930) | 60 epochs, Adam with lr 1e-3 (default betas), batch 32, seed 0 (`torch.manual_seed(0)`, shuffle generator seed 0). The epoch with the highest DEV accuracy is kept. | Earlier epoch |

- **Best baseline (for H2):** the highest DEV accuracy among the four selected models. Ties go to the model with more trainable parameters. This is decided on DEV, not on TEST_FRESH.
- **Limitation:** (c) and (d) are run with one seed each. No seed-to-seed variance is measured for them. See section 10.

## 4. Order of operations

1. D2NN DEV grid (seed 0). No TEST_FRESH evaluation.
2. Baseline selection on DEV. No TEST_FRESH evaluation.
3. Choice of the best baseline on DEV.
4. TEST_FRESH evaluation of each baseline, once.
5. D2NN final, seeds 0 to 9: DEV and TEST_FRESH, once per seed.
6. Controls: C1 (seeds 0 to 2) and C2 (seeds 0 to 9), TEST_FRESH once per seed.
7. Statistics and verdicts.

No TEST_FRESH evaluation happens before step 3 is complete. The script records each (configuration, seed) key that has been evaluated on TEST_FRESH and stops with an error if a key is evaluated a second time.

## 5. Controls

- **C1 (label permutation):** the TRAIN labels are permuted with `np.random.default_rng(100 + seed).permutation`, for seeds 0, 1 and 2. The D2NN is trained with the same configuration and evaluated on TEST_FRESH.
- **C2 (untrained network):** the D2NN with random phases from seed s, with no training, for seeds 0 to 9. Evaluated on TEST_FRESH.

## 6. Hypotheses and thresholds (fixed)

| Code | Claim | Pass criterion |
|---|---|---|
| H1 | The D2NN classifies above the accuracy threshold | Mean TEST_FRESH accuracy over seeds 0 to 9 ≥ **0.90**, and the minimum seed ≥ **0.88** |
| H2 (primary fairness test) | The D2NN is at least as accurate as the best DEV-selected baseline, and the difference is significant on seed 0 | Mean D2NN TEST_FRESH accuracy minus the best baseline's TEST_FRESH accuracy ≥ **0**, **and** exact two-sided McNemar p < **0.05** on seed-0 predictions against that baseline, **and** the seed-0 discordant pairs favour the D2NN (b > c) |
| C1 (control) | Permuted labels give no spurious learning | Mean TEST_FRESH accuracy over seeds 0 to 2 ≤ **0.15** |
| C2 (control) | An untrained readout does not classify | Mean TEST_FRESH accuracy over seeds 0 to 9 ≤ **0.20** |

**Clarification added before the run.** The third condition of H2 (b > c) is not in the wording of the request. It was added here because a significant McNemar result in the opposite direction would otherwise pass H2 with a non-negative mean difference. The condition makes H2 harder to pass and is reported as such.

## 7. Statistics

- **Per-seed Wilson 95% intervals** with n = 3823 (`optlab.stats.wilson_interval`, z = 1.96) for each D2NN seed, each baseline, and each control seed.
- **Paired statistics for each D2NN seed against the best baseline:** discordant counts b (D2NN correct, baseline wrong) and c (the reverse), and the exact two-sided McNemar p. For seeds 1 to 9 these are descriptive. The verdict uses seed 0.
- **Seed-0 McNemar against each baseline** (a) to (d), reported for all four.
- **Mean differences** (D2NN mean minus each baseline) as point estimates only.
- **No interval for any mean.** No pooled Wilson interval over seeds, no bootstrap of the mean and no interval for the mean difference. The five seeds of CONTRACT-003 share one TEST set, so a pooled binomial interval is mis-specified (CONTRACT-003, note 4).

## 8. What is always reported

- DEV and TEST_FRESH accuracy for every D2NN seed, every baseline and every control seed, with Wilson intervals.
- The DEV grids for the D2NN and for each baseline, and the selected settings.
- The paired statistics above, and the seed-0 confusion matrix of the D2NN.
- All four verdicts, whether they pass or fail.
- Compute time and peak memory. The JEV decision record.

## 9. Dry run (executed 2026-10-10, no training)

Command: `python experiments/run_fresh_test_benchmark.py --dry-run`, exit code 0, well under a minute. Output saved in the D: scratch directory. Values from the final dry run. An earlier dry run on the same machine measured 19.9 ms per gradient call, so the timings below vary by a few percent between runs.

- **Data counts:** load_digits source 1797; OpenML optdigits v1 rows 5620; source samples matched exactly 1797; TEST_FRESH 3823; TRAIN 1078; DEV 359; TEST 360 (counted, not evaluated).
- **Trainable parameters:** D2NN 49,152; (a) 40,970; (b) 650; (c) 48,985; (d) 50,618.
- **Timing on this machine:** D2NN gradient call 20.1 ms; D2NN forward pass 10.3 ms; logistic regression fit (C = 1, 64 × 64) 0.68 s; MLP fit 4.9 s; one CNN epoch 2.2 s.
- **Work in the full run:** 16 D2NN training runs (3 DEV grid, 10 final, 3 C1), that is 517,440 gradient calls and 92,596 forward passes.
- **Estimate (not a result):** about 11,500 s (3.2 h) in total. D2NN training about 10,400 s, forward passes about 950 s, the CNN (60 epochs) about 130 s, the baselines under 30 s.
- **Memory:** resident set size 565 MB at the end of the dry run, with torch loaded. The limit is about 1 GB.
- **Files:** the dry run wrote no result file. Its only write is the OpenML download cache under `D:/PROJECTS/.cognition/neuro3d-lab/cache/openml`. Nothing was written to C:.

## 10. Stated limitations

- **Small TRAIN.** TRAIN has 1078 samples, for a model with 49,152 phase parameters. The result may depend on this training-set size.
- **Shared origin.** TRAIN and DEV come from the same 1797-sample source as CONTRACT-003 TEST. TEST_FRESH is independent of the training data but not of the original dataset.
- **Writers.** If the UCI documentation holds, TEST_FRESH is writer-disjoint from TRAIN, DEV and the CONTRACT-003 TEST. This is documented, not verified per sample (section 1.1).
- **DEV is reused.** Section "Status" lists the earlier DEV use. Only TEST_FRESH is fresh.
- **Single-seed baselines.** (c) and (d) have one seed each. A baseline with larger seed variance could differ from the measured value. The D2NN, with ten seeds, has a seed spread that is measured here.
- **Grid window.** The D2NN is evaluated at N = 128, which CONTRACT-004 found to change the detector intensity by about 4% at 18 mm compared with N = 256. CONTRACT-005 found the seed-0 TEST accuracy stable at N = 256 (0.9694 against 0.9639). This contract does not repeat that comparison, so the D2NN numbers here carry that caveat.
- **Fixed learning rate.** The D2NN learning rate stays at 0.02. CONTRACT-003 also tested 0.05, and that value is not tried here.
- **Solver convergence.** Logistic regression at large C may not converge in 3000 iterations. The iteration count of each fit is recorded in the DEV grid, so a fit that reaches the 3000-iteration limit can be identified.
- **Scope.** Simulation only. No fabrication noise, misalignment, inter-plane reflections or spectral dispersion. No MNIST and no comparison with published D2NN results.

## 11. Result registry

**Status:** the full run has **not** been executed. Result: pending.

**Run record (to be completed after the run):** commit hash; start and end time (UTC); elapsed time; summary file `experiments/results/fresh_test_benchmark_summary.json` (versioned, without per-sample arrays); full record `D:/PROJECTS/.cognition/neuro3d-lab/results/fresh_test_benchmark.json` (not versioned); log `D:/PROJECTS/.cognition/neuro3d-lab/results/fresh_test_benchmark.log`.

**Verdicts (TEST_FRESH, one use per configuration and seed):** pending.

| Code | Criterion | Result | Verdict |
|---|---|---|---|
| H1 | Mean ≥ 0.90 and minimum ≥ 0.88 over seeds 0 to 9 | pending | pending |
| H2 | Mean difference to the best baseline ≥ 0; McNemar p < 0.05 and b > c on seed 0 | pending | pending |
| C1 | Mean over seeds 0 to 2 ≤ 0.15 | pending | pending |
| C2 | Mean over seeds 0 to 9 ≤ 0.20 | pending | pending |

**Reading rule.** Each verdict is reported as it falls. A failed H2 means no evidence that the D2NN is better than the best DEV-selected baseline on seed 0. It does not show that the D2NN is worse. The result does not change CONTRACT-003, CONTRACT-005 or CONTRACT-006.
