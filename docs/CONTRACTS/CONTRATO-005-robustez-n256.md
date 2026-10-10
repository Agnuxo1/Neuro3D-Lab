# CONTRATO-005: post hoc robustness check of the CONTRATO-003 configuration at N = 256

> **POST HOC.** The CONTRATO-003 main results, including TEST, were known when this contract was written. This is a **robustness check, not a confirmatory test**. Its criteria were fixed now, after those results were known, so they are a stated commitment about how the result will be read, not a pre-registration. No CONTRATO-003 verdict changes because of this contract.

**Status:** post hoc, written 2026-10-10. The dry run has been executed (no training). The full run has **not** been executed. Result: pending.
**Scripts:** `experiments/run_d2nn_robustness_n256.py` (`--dry-run` builds the model and times 20 gradient calls; without the flag it runs the full check).
**Reference:** `docs/CONTRACTS/CONTRATO-003-d2nn-digitos.md` (post-hoc note on the grid) and `docs/CONTRACTS/CONTRATO-004-regimen-d2nn.md`, T2.

## Why this check exists

CONTRACT-004, T2 measured that the intensity at 18 mm changes by a relative L2 of 4.41·10⁻² between N = 128 (the CONTRATO-003 grid) and N = 256 (threshold 2·10⁻²). Periodic wrap-around in the smaller window therefore changes the detector field by about 4 %. CONTRATO-003 states that a robustness rerun of its final configuration at N = 256 is required before any D2NN number from it is cited. This contract is that rerun.

## Question

Does the seed-0 TEST accuracy of the final CONTRATO-003 configuration change when the same geometry is run on N = 256?

## Criteria (fixed now, post hoc)

Let TEST_128 be the seed-0 TEST accuracy of CONTRATO-003 and TEST_256 the seed-0 TEST accuracy of this run.

- **Stable** if |TEST_256 − TEST_128| ≤ 0.01 **and** TEST_256 ≥ 0.85 (the H1 threshold of CONTRATO-003).
- **Unstable** otherwise.

With 360 TEST samples, the first condition is equivalent to 344 to 350 correct predictions (0.9556 to 0.9722). The second condition holds for any count of 306 or more.

These criteria do not select anything on TEST. Each accuracy is reported whatever the outcome.

## Reference values (CONTRATO-003, read from `experiments/results/d2nn_digits_summary.json` and `D:/PROJECTS/.cognition/neuro3d-lab/results/d2nn_digits.json`)

- Selected on DEV in CONTRATO-003: β = 100, lr = 0.02.
- TEST_128, seed 0: **0.9638888888888889** (347 of 360).
- TEST_128, seeds 0 to 4: 0.9639, 0.9611, 0.9583, 0.9611, 0.9667. Mean 0.9622, range 0.9583 to 0.9667.
- Seed-0 TEST confusion matrix at N = 128 is stored in the full CONTRATO-003 file and is reproduced alongside the N = 256 matrix.

## Setup

Same physical geometry, changed grid.

| Item | CONTRATO-003 (N = 128) | CONTRATO-005 (N = 256) |
|---|---|---|
| Pixel size dx | 0.4 mm | 0.4 mm (unchanged) |
| Window N·dx | 51.2 mm | 102.4 mm |
| Validity limit N·dx²/λ | 27.3 mm | 54.6 mm |
| Propagation distance | 18 mm, four times | 18 mm, four times (inside the limit) |
| Wavelength λ | 0.75 mm | 0.75 mm |
| Digit | 64 × 64 px, centred, pad 32 px per side | 64 × 64 px, centred, pad 96 px per side |
| Detector | ten 12 × 12 px regions at the same physical positions | same physical positions (144 px each) |
| Model | 3 phase layers, scale β = 100 | same |
| Training | Adam, lr 0.02, 30 epochs, batch 32 | same |
| Split | seed 2026, TRAIN 1078 / DEV 359 / TEST 360 | same split; DEV is not used at N = 256 |
| Model seed / training rng | 0 / 0 | 0 / 0 |

Encoding, unit-energy normalisation, detector masks and training are the imported functions from `experiments/run_d2nn_digits.py`, which is not modified. The N = 256 masks come from `optlab.elements.detector_masks(256, …)`. The script checks that the N = 128 masks built this way equal the CONTRATO-003 masks.

## How to read the result

- **Retraining at N = 256 is a new trajectory.** The initial phases are drawn from an array of size 3 × 256 × 256, so seed 0 does not reproduce the N = 128 initial phases, and the gradients differ. The mini-batch order is the same, because the same rng seed and the same number of samples give the same permutations. Results need not match exactly, and a difference is not by itself a failure of the model.
- **Only the window changes.** The pixel size is the same, so this check tests the periodic-window (wrap-around) effect. It does not test pixel convergence. That is CONTRACT-004 T3, which failed as written and is not rerun here.
- **One seed.** The criterion uses seed 0 alone. Across the five CONTRATO-003 seeds at N = 128 the TEST accuracy ranges over 0.0083, so a tolerance of 0.01 is about one seed-to-seed range. A stable result means the N = 256 accuracy lies within the spread already seen across seeds. It does not show that individual predictions are unchanged by the grid.
- **No re-tuning on TEST.** β = 100 and lr = 0.02 are inherited unchanged. If the result is unstable, it is reported as unstable. No hyperparameter is changed on TEST, and no further run is started to find a stable value.

## Reported regardless of outcome

- TEST_128 and TEST_256 (seed 0), their difference, and the verdict (stable or unstable).
- The per-class TEST confusion matrix (10 × 10; rows are true classes, columns predicted) for N = 256, seed 0, next to the N = 128 seed-0 matrix.
- Compute time per call and total run time.

## Not claimed

- Not a confirmatory test. The criteria were fixed after the CONTRATO-003 results were known.
- Not a multi-seed robustness result.
- Not a pixel-convergence result (CONTRACT-004 T3).
- Sampling limit: spatial frequencies with sin θ > λ/(2dx) = 0.9375 cannot be sampled on either grid, as in CONTRACT-004. This is not changed here.
- No physical experiment, fabrication noise, misalignment, inter-plane reflection or spectral dispersion.

## Run procedure

1. `python experiments/run_d2nn_robustness_n256.py --dry-run` (no training, no files). Exit code 0 on 2026-10-10. Estimated full run: about 65 min on the machine where it was executed (118 ms per gradient call at N = 256). This is an estimate, not a result.
2. Full run `python experiments/run_d2nn_robustness_n256.py`. Run in the background. It only starts after the coordinator approves it.
3. Outputs, written only by the full run:
   - `D:/PROJECTS/.cognition/neuro3d-lab/results/d2nn_robustness_n256.json` (full, not versioned; includes per-sample TEST predictions and labels).
   - `experiments/results/d2nn_robustness_n256_summary.json` (versioned; no per-sample arrays).
   - `D:/PROJECTS/.cognition/neuro3d-lab/results/d2nn_robustness_n256.log`.

## Result registry

| Quantity | Value | Verdict |
|---|---|---|
| TEST_128, seed 0 (CONTRATO-003) | 0.9638888888888889 (347/360) | reference |
| TEST_256, seed 0 | pending (full run not executed) | pending |
| \|TEST_256 − TEST_128\| ≤ 0.01 | pending | pending |
| TEST_256 ≥ 0.85 | pending | pending |
| Verdict | pending | pending |
