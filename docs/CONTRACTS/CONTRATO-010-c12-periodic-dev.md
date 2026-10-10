# CONTRATO-010 (POST HOC, DEV only, not a verdict on TEST)

**Status:** post hoc. Written 2026-10-10 **before** any run of `experiments/c12_periodic_dev_d2nn.py`. The file is uncommitted at the time of writing; its commit hash must be recorded here before the run. TEST is not used. This contract does not change CONTRATO-003 (its verdicts stand), does not implement the C12 fix, and does not re-run any D2NN experiment.

**Question.** How much does the periodic 51.2 mm FFT window, the window used by the CONTRATO-003 propagation, change the DEV predictions of the seed-0 D2NN? The comparison is with a zero-padded propagation on a 102.4 mm window.

**Background.** CONTRATO-006c (post hoc) found that, for a tilted input, the periodic FFT window alone accounts for a 13 % full-grid error of the angular spectrum (annulus 0.16). The effect on the D2NN itself (N = 128, dx = 0.4 mm, d = 18 mm) was not quantified (ROADMAP, C12). This contract measures it on DEV only, with the thresholds fixed in advance.

## Fixed setup

- **Prerequisite.** `experiments/save_d2nn_seed0.py` must be run first. It reproduces the CONTRATO-003 seed-0 DEV accuracy (0.9554317548746518) before the phases are used. The phases file `experiments/results/d2nn_seed0_phases.npz` is produced by that script; it does not exist yet, so no hash of it is recorded here.
- **Model.** Seed-0 phases from `experiments/results/d2nn_seed0_phases.npz`, shape (3, 128, 128). They are written by `experiments/save_d2nn_seed0.py`, which retrains seed 0 on TRAIN only (beta = 100.0, lr = 0.02) and checks that DEV accuracy equals 0.9554317548746518 within 1e-12. The phases are not modified by this contract.
- **Architecture.** As in CONTRATO-003: N = 128, dx = 0.4 mm, λ = 0.75 mm, DIST = [18 mm] × 4 (input → layer 1 → layer 2 → layer 3 → detector). Detector masks from `run_d2nn_digits.build_masks()`: ten 4.8 mm square regions.
- **Data.** The 359 DEV images of the split with SPLIT_SEED = 2026, encoded with `run_d2nn_digits.encode_field` (identical to CONTRATO-003). TEST indices are returned by the split helper and discarded. They are not encoded, not loaded for evaluation and not evaluated.
- **Propagator.** `optlab.propagation.angular_spectrum` for both propagations. `optlab` is not changed.

## Computed propagations

- **(P) periodic.** The registered propagation: the 128 × 128 grid, a 51.2 mm periodic window. It is computed with the registered `DiffractiveNetwork.forward`.
- **(Z) zero-padded.** A 256 × 256 grid (window 102.4 mm). The 128 input field is placed at offset 64 (indices 64 to 191) and the zero field elsewhere. The three phase masks are placed at the same offset, with phase 0 (transparent) outside the 128 window. The ten detector masks are embedded at the same offset, so the physical coordinates of the two grids agree. Intensities are read from the central 128 window. Distances, dx and λ are unchanged.
- **Scores and prediction.** For each image, s_k = beta × Σ(mask_k × I), k = 0 to 9, and the predicted class is argmax_k s_k. This is the rule of `DiffractiveNetwork.predict`.

## Pre-registered quantities and thresholds (fixed before any run)

| ID | Quantity | Negligible | Material |
|---|---|---|---|
| Q1 | DEV accuracy of P reproduces 0.9554317548746518 | gate: \|acc_P − 0.9554317548746518\| ≤ 1e-9; otherwise the run is invalid and no verdict is written | — |
| Q2 | \|acc_P − acc_Z\| on DEV | ≤ **0.005** | ≥ **0.02** |
| Q3 | fraction of DEV images with argmax_P ≠ argmax_Z | ≤ **0.01** | ≥ **0.03** |
| Q4 | median over DEV images of ‖s_Z − s_P‖₂ / ‖s_P‖₂ (10 scores) | ≤ **0.01** | ≥ **0.05** |

A value between the negligible and material bounds is classed as **intermediate**. Descriptive values (p90 and maximum of the Q4 ratio, counts of images correct under only one propagation) are reported but do not enter the outcome.

## Decision rule

1. **Q1 gate.** If Q1 fails, the run is invalid. The script prints the failure, exits with status 2 and writes no verdict and no output file. No other quantity is read.
2. **Material** if any of Q2, Q3, Q4 is material.
3. **Negligible** if all of Q2, Q3, Q4 are negligible.
4. **Inconclusive** otherwise. It is reported with its values.

## Scope and consequences

- **Material.** The DEV predictions of the registered D2NN depend on the periodic window at these thresholds. The C12 zero-padded propagation must then be applied, and the CONTRATO-003 results recomputed on it, before any D2NN result is published (ROADMAP C12). This contract does not itself re-run TEST or CONTRATO-003.
- **Negligible.** At these thresholds, the periodic window does not change the DEV predictions of seed 0. This does not show that the periodic result is physically correct. The zero-padded grid is itself finite and has its own images at 102.4 mm, so Z measures sensitivity to the window, not the error of either propagation.
- **Post hoc.** The comparison was chosen after CONTRATO-006c. The thresholds above are fixed here, before the run.
- **Limits.** Seed 0 only. Scalar model only; vector effects are not tested. The input is the digit field of CONTRATO-003, not the tilted wave of CONTRATO-006c, so the 13 % figure does not transfer to this input. No prediction of the size of the effect is made here.
- **Provenance.** The script records the SHA-256 of the phases file it uses, and whether it matches `d2nn_seed0_checksum.json` when that file is present.
- **Outputs.** Script: `experiments/c12_periodic_dev_d2nn.py`. Result: `experiments/results/c12_periodic_dev_d2nn.json`, written only if Q1 passes. Nothing is written to drive C:. No TEST number is produced.

## Results (POST HOC, to be filled after the run)

- Q1 (periodic DEV accuracy): to be filled after the run.
- Q2, Q3, Q4 values and classes: to be filled after the run.
- **Outcome:** to be filled after the run.
