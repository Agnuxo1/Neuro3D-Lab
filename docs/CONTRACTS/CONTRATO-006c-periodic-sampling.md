# CONTRATO-006c (POST HOC diagnostic, not a verdict)

**Status:** post hoc. Written 2026-10-10 **before** any run of `experiments/diagnose_v2_periodic_aliasing.py`. It does not replace or change the CONTRATO-006 verdict: V2 remains **fail**.

**Question.** The full-grid discrepancy between the angular spectrum (ASM) and the Rayleigh–Sommerfeld (RS) reference for the tapered tilted wave I2 is 0.16 in the annulus 12.8 < r ≤ 25.6 mm (CONTRATO-006b, addendum 2). Two mechanisms can produce it, and this contract separates them:

- **(A) Periodic images.** The ASM is an FFT propagator. Its 51.2 mm window is periodic, so energy that leaves the window on the +x side (lateral shift about 10.4 mm for sin θ = 0.498 over d = 18 mm) re-enters on the −x side. The RS reference has no such images.
- **(B) Input sampling (aliasing).** The ASM sees the input only at pitch dx = 0.4 mm. The RS reference samples the continuous input at pitch dx/2.

**Why this order.** Addendum 2 of CONTRATO-006b refuted the finite-window hypothesis. The RS 51.2 mm and RS 205 mm references agree to 1.2 × 10⁻⁹. Mechanism (B) was the next hypothesis in the handoff, but it was not the only candidate. Mechanism (A) follows from the periodic FFT itself, and it is the one the ROADMAP lists as an unimplemented fix (C12).

## Fixed setup (identical to CONTRATO-006 and 006b)

- Input I2: tilted wave exp(i k sin θ x) with sin θ = 0.498047 (DFT-bin tilt, 34 bins over 51.2 mm), multiplied by the super-Gaussian taper exp[−(ρ/18 mm)⁸]. λ = 0.75 mm, d = 18 mm, k = 2π/λ.
- Output nodes: the 128 × 128 grid at dx = 0.4 mm, node positions (n − 63.5)·dx.
- The reference is RS with a 51.2 mm source window at source pitch dx/2 (256 × 256 sources), the CONTRATO-006 V2 reference. It equals RS 205 mm on the full grid (CONTRATO-006b).
- Metric: relative L2 error of the complex field over all 16 384 nodes, and in the three radial bands of 006b (r ≤ 6.4, 6.4 < r ≤ 12.8, 12.8 < r ≤ 25.6 mm).

## Computed fields

- **R1 ASM128.** `optlab.propagation.angular_spectrum` on the 128 grid at dx. This is the V2 ASM field. Its relative error against R2 must reproduce 0.1335 (check).
- **R2 RS51.** The CONTRATO-006 reference described above (the reference).
- **R3 ASM256-pad.** The same taper and tilt sampled at dx on a 256 × 256 grid (window 102.4 mm). Compare its central 128 × 128 nodes (indices 64–191) with the R1 nodes. The window is large enough that the field outside ±25.6 mm is below 1e-8, so periodic images are negligible.
- **R4 ASM-fine.** The same taper sampled at dx/2 on a 256 × 256 grid whose window is 51.2 mm, so it has the same period and the same periodic images as R1. Node positions (m − 128)·dx/2. Compare its odd nodes m = 2n + 1, which coincide with the R1 node positions, with R1 and R2.

Only `optlab` functions are called. The frozen modules (`d2nn.py`, `propagation.py`, `elements.py`, `fresnel.py`) are not changed.

## Pre-registered quantities and thresholds (fixed before the run)

| Quantity | Definition | Tolerance |
|---|---|---|
| Q1 | rel(R1, R2), full grid | reproduces 0.1335 (check, ± 1e-6) |
| Q2 | rel(R3 central, R2), full grid and annulus | images explain V2 if ≤ **1e-2** (the V2 tolerance); images refuted if ≥ **5e-2** |
| Q3 | rel(R4 odd nodes, R2), full grid and annulus | sampling explains V2 if ≤ **1e-2**; sampling refuted if ≥ **5e-2** |
| Q4 | rel(R1, R3 central) | decomposition only: periodic-image effect on the ASM itself |
| Q5 | rel(R1, R4 odd nodes) | decomposition only: input-sampling effect at identical windows |

## Decision rule

1. **Images explain the discrepancy** if Q2 (full grid) ≤ 1e-2 and Q2 (annulus) ≤ 1e-2.
2. **Sampling explains the discrepancy** if Q3 (full grid) ≤ 1e-2 and Q3 (annulus) ≤ 1e-2.
3. **Both** if both conditions hold. This is reported as an inconsistency to resolve, not as a joint explanation.
4. **Neither** if Q2 and Q3 are both ≥ 5e-2 on the annulus. The discrepancy then points to the scalar kernel or the ASM model, which is an open hypothesis, not a result.
5. Any other pattern is **inconclusive** and is reported with its values.

Q1 must reproduce before any other quantity is read. If it does not, the run is invalid and this contract is not evaluated.

## Scope and consequences

- The result is post hoc and does not change CONTRATO-006 V2 (fail).
- The D2NN experiments (CONTRATO-003 to 005, 007) use the periodic FFT on a 51.2 mm window at dx = 0.4 mm. If decision rule 1 holds, those propagations carry a boundary artefact, and the grid fix (zero-padded propagation, C12) must be done before any D2NN result is published. This diagnostic does not itself re-run those experiments.
- If decision rule 2 holds, the input-sampling regime (CONTRATO-004) is the relevant limit and must be stated in the README.
- Neither result tests vector effects. The diagnostic is a scalar-model test only.
- Output: `experiments/results/diagnose_v2_periodic_aliasing.json` (repository). Log: `D:/PROJECTS/.cognition/neuro3d-lab/tmp-diag/v2_periodic_aliasing.log`.
