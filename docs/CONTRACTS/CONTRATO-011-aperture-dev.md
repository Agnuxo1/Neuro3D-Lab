# CONTRATO-011 (POST HOC, DEV only, not a verdict on TEST)

**Status:** post hoc. Written 2026-10-10 **before** any run of `experiments/c12b_aperture_dev_d2nn.py`. The file is uncommitted at the time of writing; its commit hash must be recorded here before the run. TEST is not used. This contract does not change CONTRATO-003 or CONTRATO-010, and does not re-run any D2NN training.

**Question.** CONTRATO-010 found that the seed-0 D2NN DEV accuracy falls from P = 0.9554317548746518 (periodic 51.2 mm window) to Z = 0.8495821727019499 (zero-padded, with phase layers transparent outside the 128 window). Does that drop depend on the assumption that the phase layers are transparent outside the window?

## Fixed setup

Identical to CONTRATO-010: seed-0 phases from `experiments/results/d2nn_seed0_phases.npz` (shape (3, 128, 128)), beta = 100.0, the 359 DEV images (SPLIT_SEED = 2026) encoded with `run_d2nn_digits.encode_field`, the same detector masks from `run_d2nn_digits.build_masks()`, N = 128, dx = 0.4 mm, lambda = 0.75 mm, DIST = [18 mm] x 4. TEST indices are discarded, never encoded and never evaluated. Propagator: `optlab.propagation.angular_spectrum`; `optlab` is not changed.

The only change is the propagation convention.

## Convention A (aperture)

- The field lives on a 256 x 256 grid; the 128 input field is centred at offset 64.
- After each propagation (including the last one), the field is multiplied by the 128 x 128 central window mask (1 inside, 0 outside). Light that leaves the window is lost; it does not wrap and does not re-enter.
- The phase layers act only inside the 128 window (exp(i phi) inside; the field is zero outside anyway).
- Detector intensities are read from the central 128 window; scores are s_k = beta x sum(mask_k x I), prediction = argmax_k s_k.

## Pre-registered rule (fixed before any run)

Let A be the DEV accuracy under convention A, P = 0.9554317548746518 (periodic) and Z = 0.8495821727019499 (zero-padded, transparent outside).

- **R1.** If |A - P| <= 0.01, the drop in Z is caused by the transparent-outside assumption, and the periodic result is robust to a physical aperture.
- **R2.** If |A - Z| <= 0.01, the drop is caused by light leaving the window and re-entering or being recovered by the transparent layers, i.e. the periodic wrap-around is the operative effect in both padded conventions.
- **R3.** Otherwise the result is inconclusive.

(P and Z differ by 0.106, so R1 and R2 cannot both hold; if the script ever reports both, it labels the outcome ambiguous.)

Also reported: the fraction of DEV predictions that change between P and A, and between Z and A (Z is recomputed per image by the script and checked against 0.8495821727019499 within 1e-9).

**Q1 gate.** The periodic DEV accuracy must reproduce P within 1e-9. Otherwise the run is invalid; the script prints the failure, exits with status 2, and writes no verdict and no output file.

## Scope and limits

- DEV only, seed 0, scalar model. A is one physical aperture model (hard cutoff, no edge diffraction from a finite-thickness stop, no absorbing boundary beyond the 128 window); other aperture models are not tested.
- Post hoc: chosen after CONTRATO-010; the rule above is fixed here, before the run.
- Nothing is written to drive C:. No TEST number is produced.
- Outputs: script `experiments/c12b_aperture_dev_d2nn.py`; result `experiments/results/c12b_aperture_dev_d2nn.json`, written only if Q1 passes.

## Results (POST HOC, to be filled after the run)

- Q1 (periodic DEV accuracy): to be filled after the run.
- A (aperture DEV accuracy), |A - P|, |A - Z|: to be filled after the run.
- Fraction of predictions changed, P vs A and Z vs A: to be filled after the run.
- **Outcome (R1, R2 or R3):** to be filled after the run.
