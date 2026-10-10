# CONTRACT-006: direct-summation Rayleigh-Sommerfeld reference at dx/λ = 0.53

## Status and provenance

- **Gap closed by this contract:** G01 (P0), independent reference for the propagator in the D2NN sampling regime (ROADMAP §2.1).
- **Pre-registration:** written 2026-10-10 (UTC) before any full run. The thresholds are the ones fixed in the request and are **not changed**. The sampling, wavelength, pitch, window and distance are also unchanged.
- **Amendments A1 to A3 (below):** the request specified an RS kernel form, an S0 gate and an I2 definition. Feasibility checks at 32 × 32 and 128 × 128 on scratch data (no full run, no reference results) showed that the specified forms cannot pass their own gates, for reasons that are analytic or geometric rather than statistical. The amendments were made before any full run. Probe scripts: `D:/PROJECTS/.cognition/neuro3d-lab/tmp-rs006/`.
- **Decision record:** the amendment decisions were taken with the JEV router (`router.py jev`, `provenance=jev`, `remote_decision=true`). Confidences: kernel 1.00, S0 gate 1.00, I2 definition 1.00, documented amendment policy 0.98. JEV's estimate that the specified S0 can pass as written was 0.05.
- **Status:** full run **not executed**. The S0 gate (sanity) was executed; its result is in the registry at the end of this file.

## Question

Does the angular-spectrum propagator (`optlab.propagation.angular_spectrum`) agree with an independent direct summation of the Rayleigh-Sommerfeld integral for the fields and detector intensities used in CONTRACT-003 to CONTRACT-005, at dx/λ = 0.53 and at a distance inside the ASM validity limit?

## 1. Physical setup (fixed)

| Symbol | Value | Note |
|---|---|---|
| λ | 0.75 mm | vacuum; medium index 1, so k = 2π/λ |
| dx | 0.4 mm | coarse pitch; dx/λ = 0.53 |
| N | 128 | coarse grid; window N·dx = 51.2 mm |
| d | 18 mm | propagation distance (24 λ) |
| z_max (ASM) | N·dx²/λ = 27.3 mm | d < z_max, so the ASM is inside its validity domain |

Coordinates follow `optlab.elements`: x_j = (j − (N−1)/2)·Δ on columns, y_i = (i − (N−1)/2)·Δ on rows, with Δ = dx for the coarse grid. The reference sources are on the same physical window at Δ = dx/2 (N_s = 256) and, for C1 only, Δ = dx/4 (N_s = 512). Midpoint quadrature: dA = Δ².

## 2. Reference: exact first Rayleigh-Sommerfeld kernel and normalisation

For a scalar field with time dependence exp(−iωt), propagating in +z, the exact first RS solution is

U(P) = −(1/2π) ∫∫ U₀(Q) ∂_z(e^{ikr}/r) dA = (1/(iλ)) ∫∫ U₀(Q) · (z/r) · (1 + i/(kr)) · e^{ikr}/r · dA,

with r = |P − Q| and z = d. The two forms agree because ∂_z(e^{ikr}/r) = ik (z/r)(1 + i/(kr)) e^{ikr}/r, and −ik/(2π) = 1/(iλ).

The specified form, (1/(iλ)) (z/r) e^{ikr}/r without the factor (1 + i/(kr)), is the leading term in 1/(kr). It is kept as diagnostic K_spec (D4 to D6). The reference in this contract is K_exact.

**Normalisation check, infinite plane, uniform unit input, on axis.** Let J = ∫_z^∞ e^{ikr}/r dr = E₁(−ikz). Then:

- K_spec gives I_spec = −ikz·J. Its asymptotic expansion is e^{ikz}(1 − i/(kz) + …). At kd = 150.80, |I_spec − e^{ikd}| = 6.63 × 10⁻³ (= 1/(kd)). This is **above the 1e-3 threshold**, so the specified kernel cannot meet S0 even with an infinite aperture.
- K_exact adds (i/k)∫_z^∞ e^{ikr}/r² dr = (i/k)(e^{ikz}/z + ikJ), which gives I_exact = −ikz[J + (i/k)(e^{ikz}/z + ikJ)] = e^{ikz} **exactly**. The closed-form numerical error is 1.9 × 10⁻¹⁹.

So the prefactor 1/(iλ) is confirmed and the leading-order kernel carries the error 1/(kd).

## 3. Amendments to the form specified in the request

| Item | As specified | Amended (this contract) | Measured reason (scratch) |
|---|---|---|---|
| **A1 kernel** | (1/(iλ)) e^{ikr}/r cos θ | K_exact = K_spec · (1 + i/(kr)) = −(1/2π)∂_z(e^{ikr}/r); K_spec reported as diagnostic | Infinite-plane error 6.63 × 10⁻³ > 1e-3 (section 2). Disc S0 with K_spec: 1.24 × 10⁻² |
| **A2 S0 gate** | uniform unit input on the square 32 × 32 window, on-axis value compared with e^{ikd} | uniform unit input on the disc ρ ≤ R = N_S·dx/2 at source pitch dx/2, on-axis value compared with the closed form in section 4, threshold 1e-3 unchanged. The square-window value vs e^{ikd} is reported as D7 | A finite window truncates the uniform field. The edge term (z/r_R)e^{ikr_R} is physical, not a kernel error. Square 32 × 32 gives 0.39 for either kernel (threshold 1e-3). Disc closed form gives 6.2 × 10⁻⁴ |
| **A3 I2 frequency** | sin θ = 0.5 | sin θ_b = 34·λ/(N·dx) = 0.498047 (nearest DFT bin on the grid). Untapered, ASM vs exact tilted wave is reported as D1 | sin θ = 0.5 is not a DFT bin on N = 128 (k_x·N·dx/2π = 34.13). ASM error vs exact = 0.37. Bin-aligned: 2.7 × 10⁻¹⁴ |
| **A3 I2 for V2** | untapered plane wave | I2 multiplied by a super-Gaussian taper T(ρ) = exp[−(ρ/ρ₀)⁸], ρ₀ = 18 mm; the untapered wave is kept for D1 to D3 | Untapered RS truncation: RS vs ASM 5.4 × 10⁻² (central 8 × 8 points), above the 1e-2 threshold. Tapered: 4.2 × 10⁻⁷ |

Thresholds, λ, dx, N, d, the test-input families I1 and I3, and the metrics V1, V3 and C1 are unchanged.

## 4. Gate S0: closed form for the disc window

Let R = N_S·dx/2 (N_S = 32 for the gate, so R = 6.4 mm). Sample the disc ρ ≤ R on the fine grid (pitch dx/2) and take R_eff = √(n_in·(dx/2)²/π), where n_in is the number of sampled points. Define r_R = √(R_eff² + d²).

For a uniform unit input on this disc, with K_exact:

U_disc = −∂_z ∫₀^R ρ e^{ikr}/r dρ = −∂_z[(e^{ikr_R} − e^{ikz})/(ik)] = e^{ikz} − (z/r_R) e^{ikr_R}.

The term (z/r_R)e^{ikr_R} is the physical truncation term. It is 0.94 in magnitude at R = 6.4 mm and 0.58 at R = 25.6 mm, so no finite window reproduces e^{ikd} on axis to 1e-3. The S0 gate therefore compares the numerical sum with U_disc.

**S0 gate.** |U_RS − U_disc| ≤ 1e-3, with U_RS evaluated on axis at (0, 0) for the disc input at dx/2 sampling, N = 32, d = 18 mm, K_exact. Values from scratch: 6.18 × 10⁻⁴ at dx/2, 4.74 × 10⁻⁴ at dx/4, 1.95 × 10⁻⁵ at dx/8.

**Scope note.** The same disc test on N = 128 gives 1.25 × 10⁻³ at dx/2 and 1.25 × 10⁻⁴ at dx/4, because of the staircase boundary of the disc. The full-run convergence test is therefore C1 on the smooth I1 input, not S0 on a disc.

## 5. Test inputs (fixed)

- **I1 (smooth Gaussian):** u₀ = exp(−(x² + y²)/w²), w = 3.0 mm, centred. Edge value at the window boundary exp(−(25.6/3)²) ≈ 2 × 10⁻³².
- **I2 (tilted wave, amended):** u₀ = T(ρ) · exp(i k sin θ_b x), sin θ_b = 34λ/(N·dx) = 0.498047 (cos θ_b = 0.867), T(ρ) = exp[−(ρ/18 mm)⁸]. Edge value of T at the window boundary ≈ 5 × 10⁻⁸. The untapered version (T = 1) is used only in D1 to D3.
- **I3 (digit, CONTRACT-003 encoding):** first sample of class 3 in `sklearn.datasets.load_digits`, a = img/16 ∈ [0, 1] (8 × 8). Coarse: np.kron(a, ones(8,8)) (64 × 64), zero-padded to 128 × 128 (pad 32 each side). Fine: np.kron(a, ones(16,16)) (128 × 128), zero-padded to 256 × 256 (pad 64 each side). Both are divided by the same constant ‖kron(a, ones(8,8))‖₂, so Σ|u_c|² = 1 on the coarse grid and the physical energy is equal on both samplings. This is the same geometry as the D2NN input. Digit encoding idea copied from `experiments/run_regime_d2nn.py`; no import.

## 6. Metrics and thresholds

Intensity relative L2 is ‖I_a − I_b‖/‖I_b‖ with I = |u|² on the coarse grid. Complex-field relative L2 is ‖u_a − u_b‖/‖u_b‖. The reference is b in every case: RS at dx/2 for V1 to V3, RS at dx/4 for C1.

| Code | Quantity | Threshold | Verdict rule |
|---|---|---|---|
| **S0** | \|U_RS − U_disc\| on axis, disc input, N = 32, pitch dx/2, K_exact (absolute, unit input) | ≤ **1e-3** | gate; a full run does not start unless S0 passes |
| **C1** | I1 intensity, RS(dx/2) against RS(dx/4) | ≤ **5e-3** | reference convergence |
| **V1** | I1 intensity, ASM against RS(dx/2) | ≤ **1e-2** | |
| **V2** | I2 (tapered) complex field, ASM against RS(dx/2) | ≤ **1e-2** | |
| **V3** | I3 intensity, ASM against RS(dx/2) | ≤ **5e-2** | |

**Diagnostics (reported regardless of outcome, not gates):**

- D1: untapered bin-aligned I2, ASM against the exact tilted wave exp(i(k sin θ_b x + k cos θ_b d)). Expected to be at machine precision (measured 2.7 × 10⁻¹⁴ on N = 128).
- D2: untapered I2, RS against the exact tilted wave (truncation; measured 5.5 × 10⁻² on the central 8 × 8).
- D3: untapered I2, ASM against RS (truncation; measured 5.4 × 10⁻² on the central 8 × 8).
- D4: I1 intensity, K_spec against K_exact (effect of the kernel form at full scale).
- D5: I1 intensity, ASM against RS with K_spec.
- D6: S0 with K_spec on the disc.
- D7: S0 with the square 32 × 32 window against e^{ikd} (measured 0.39).
- D8: infinite-plane closed forms (section 2).
- Also the S0 values at dx/4 and dx/8 (quadrature convergence).

## 7. Sampling, cost and memory

- The RS kernel is evaluated once per (observation, source) pair and applied to all input columns together, for both kernel forms. Observation points are chunked so that each chunk holds at most 4 × 10⁶ complex kernel entries (about 64 MB per array, under 1 GB in total).
- Kernel evaluations: dx/2 reference, 128² × 256² = 1.07 × 10⁹; dx/4 reference for C1, 128² × 512² = 4.29 × 10⁹. Total about 5.4 × 10⁹ complex exponentials. The full run is **not** executed in this contract's first phase.
- ASM: `optlab.propagation.angular_spectrum` on the coarse grid, N = 128, d < z_max.

## 8. Interpretation rules and scope

- All values are reported regardless of outcome. A failure of V3 is informative about the discrete scalar model at this sampling (dx/λ = 0.53, d = 24 λ, N = 128); it does not by itself refute the D2NN results.
- If C1 fails, the RS reference is not converged at dx/2 and V1 to V3 cannot be interpreted as ASM validation. This will be stated in the report.
- The S0 gate passes only if the reference reproduces the closed-form disc field to 1e-3. S0 does not test the ASM.
- **Not covered:** vector effects and polarisation, reflections between planes, spectral dispersion, fabrication noise, and hardware. The reference is a scalar RS model, and the ASM it checks is the scalar model used by the D2NN code path. This contract does not test a vector solver (ROADMAP G01 also asks for one).

## 9. Run protocol

1. `python experiments/run_rayleigh_sommerfeld.py --sanity` (S0 gate only). Exit code 0 means S0 passes. This is the only run executed in this phase.
2. `python experiments/run_rayleigh_sommerfeld.py` (full run). It repeats S0 and stops if it fails. Outputs: `experiments/results/rayleigh_sommerfeld_summary.json` (values and verdicts only) and `$NEURO3D_LAB_OUT/results/rayleigh_sommerfeld.json` (full record, default `D:/PROJECTS/.cognition/neuro3d-lab/results/`), plus a field archive `rayleigh_sommerfeld_fields.npz` in the same directory. The full run is to be started separately.

## 10. Registry

**Sanity run (S0 gate only), 2026-10-10 (UTC).** Command: `python experiments/run_rayleigh_sommerfeld.py --sanity`. No output files written.

| Quantity | Value | Threshold | Verdict |
|---|---|---|---|
| S0 normalisation error, disc window (R_eff = 6.4109 mm, N = 32, pitch dx/2, K_exact) | 6.179 × 10⁻⁴ | ≤ 1e-3 | **pass** |
| Exit code | 0 | | |

Diagnostics from the same run (not gates): D1 infinite-plane K_spec error 6.630 × 10⁻³; D2 infinite-plane K_exact closed-form error 1.941 × 10⁻¹⁹; D3 disc S0 at dx/4 4.744 × 10⁻⁴ and at dx/8 1.949 × 10⁻⁵; D4 disc S0 with K_spec 1.241 × 10⁻²; D5 square window vs e^{ikd} 3.907 × 10⁻¹.

**Full run:** not executed. The gate for starting it is S0 passing, which it does. C1, V1, V2 and V3 have not been evaluated.

## Result registry

Run: `python experiments/run_rayleigh_sommerfeld.py`, completed 2026-10-10 (about 06:20 local). Full record outside the repository; summary in `experiments/results/rayleigh_sommerfeld_summary.json`.

| Code | Quantity | Measured | Threshold | Verdict |
|---|---|---|---|---|
| S0 | normalisation error, disc window, K_exact | 6.18 × 10⁻⁴ | ≤ 1 × 10⁻³ | **pass** |
| C1 | I1 intensity, RS(dx/2) against RS(dx/4) | 3.8 × 10⁻¹⁵ | ≤ 5 × 10⁻³ | **pass** |
| V1 | I1 intensity, ASM against RS(dx/2) | 1.6 × 10⁻¹⁵ | ≤ 1 × 10⁻² | **pass** |
| V2 | I2 tapered complex field, ASM against RS(dx/2) | 1.33 × 10⁻¹ | ≤ 1 × 10⁻² | **fail** |
| V3 | I3 digit intensity, ASM against RS(dx/2) | 4.48 × 10⁻² | ≤ 5 × 10⁻² | **pass** (marginal) |

**Interpretation, stated without post hoc changes.**
- C1 and V1 pass to machine precision, but I1 is a smooth Gaussian: its integrand is so well resolved that both the reference and the angular spectrum are essentially exact for it. These two tests therefore do not discriminate between the methods. Their passing does not validate the discrete model at this sampling.
- V2 fails by a large margin. The pre-registered metric attributes the whole discrepancy to the angular spectrum, but the test did not check the reference's own convergence for the oscillatory input I2. The most likely explanation, not yet tested, is quadrature error of the reference: at source pitch dx/2 the tilted wave is sampled about 3.75 times per wavelength, so the direct sum is not spectrally converged for I2. The check that would settle it is C1 applied to I2 (RS at dx/2 against RS at dx/4), which was not part of this pre-registration. It is planned as a labelled follow-up, not as a repair of CONTRATO-006.
- V3 passes with a margin of 0.005. The digit input has sharp edges, which both methods sample at the same resolution.

**Consequence.** The independent reference at dx/λ = 0.53 is not yet established: the pre-registered validation of the angular spectrum against Rayleigh–Sommerfeld failed on the oscillatory test (V2). The digit-intensity agreement (V3) passes, but it is marginal and does not by itself establish the reference.
