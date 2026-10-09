# CONTRACT-004: propagation in the D2NN sampling regime (dx/λ ≈ 0.53)

**Status:** pre-registered on 2026-10-10 (local time), before running `experiments/run_regime_d2nn.py`. Reference commit: the one that adds this file.
**Motivation:** the independent audit (T-001 / gap G01–G03) found that CONTRATO-002 validated the propagator only at dx/λ between 3 and 13. The D2NN experiment (CONTRATO-003) uses dx/λ = 0.53. This contract tests that regime.
**Rule:** thresholds are fixed here and are not relaxed after measuring. Failures are published with their values.

## Setup (shared)

- λ = 0.75 mm, pixel dx = 0.4 mm (dx/λ = 0.533), grid N = 128 (window 51.2 mm), distance d = 18 mm. Same geometry as CONTRATO-003.
- Validity domain for plain angular spectrum: d < N·dx²/λ = 27.3 mm. The distance d = 18 mm is inside it.
- Sampling limit: the grid carries spatial frequencies up to 1/(2·dx) = 1.25 mm⁻¹. The propagating band reaches 1/λ = 1.33 mm⁻¹. Plane waves with sin θ > λ/(2 dx) = 0.9375 (θ > 69.5°) cannot be sampled on this grid. This limit is a fact of the setup and is reported, not tested.

## T1: tilted plane waves, exact solution (analytic)

- Input u0(x, y) = exp(i·2π·f_k·x), with f_k = k / (N·dx), k = 0, …, 63 (grid frequencies that are exactly periodic).
- Exact output: exp(i·2π·(f_k·x + d·sqrt(1/λ² − f_k²))).
- Metric: relative L2 error of the complex field, ‖U_num − U_exact‖ / ‖U_exact‖.
- **Threshold:** max over k = 0…63 of the relative error ≤ 1e-10.
- Reported: the error per k, and the largest angle sin θ_k = λ·f_k.

## T2: periodic boundary, D2NN input (zero-padding convergence)

- Input: the 8×8 digit (first sample labelled 3 in `load_digits`), amplitude img/16, upsampled ×8, normalized to unit energy, placed at the center.
- Compare the intensity after d = 18 mm computed on N = 128 (the D2NN grid) and on N = 256 (same dx, zero-padded to double the window), restricted to the central 128×128 region.
- Metric: relative L2 difference of intensity, ‖I_128 − I_256‖ / ‖I_256‖.
- **Threshold:** ≤ 0.02.

## T3: pixel convergence, D2NN input

- Same physical digit (25.6 mm wide), same physical window of 102.4 mm, both grids inside the validity domain.
- Reference: dx = 0.4 mm, N = 256 (validity limit 54.6 mm). Fine: dx = 0.2 mm, N = 512 (validity limit 27.3 mm). The digit is upsampled to the same physical size on both grids.
- The fine intensity is block-averaged 2×2 to dx = 0.4 mm. Both are compared on the central 51.2 mm (128×128 coarse pixels) so that only the pixel size differs.
- Metric: relative L2 difference of intensity.
- **Threshold:** ≤ 0.05.

### Amendment 1 (dated 2026-10-10, before any execution)

The first draft of T3 used N = 256 with dx = 0.2 mm (window 51.2 mm). Its validity limit is N·dx²/λ = 13.7 mm, below the 18 mm propagation distance, so that comparison would have run outside the stated domain. The design above replaces it. The thresholds are unchanged, and no result had been computed when this amendment was made.

## What is not claimed by this contract

- Agreement with an independent reference solver (Rayleigh–Sommerfeld direct summation, FDTD): not tested here. Remains an open gap (G01, P0).
- Vector effects, polarization, or any statement about fabricated devices.
- Results beyond the single wavelength and the single geometry above.

## Result registry

Run: `python experiments/run_regime_d2nn.py` (about 8 s on CPU), commit `15ce4f0`, date 2026-10-10.

| Test | Pre-registered metric | Measured | Threshold | Verdict |
|---|---|---|---|---|
| T1 | max relative error over k = 0…63 | 3.7·10⁻¹⁴ (worst k = 63, sin θ = 0.923) | ≤ 1·10⁻¹⁰ | **pass** |
| T2 | relative L2 of intensity, N = 128 vs N = 256 at 18 mm | 4.41·10⁻² | ≤ 2·10⁻² | **fail** |
| T3 | relative L2 of intensity, block mean (as written) | 3.04 | ≤ 5·10⁻² | **fail** (specification error, see below) |

**Interpretation.**
- T1 confirms that the implementation reproduces the exact plane-wave solution at every frequency the D2NN grid can represent. It tests the implementation, not the physics model.
- T2 is a real finding. Periodic wrap-around in the N = 128 grid used by CONTRATO-003 changes the intensity at the detector by about 4 % (relative L2) compared with N = 256. The D2NN results must therefore be checked with a larger grid before they are cited. This is an open item, not a settled result.
- T3 failed because of the wording of the contract. "Block-averaged" is wrong for fields normalised to unit energy per grid: a coarse pixel holds the **sum** of its four fine pixels, and the mean differs from it by a factor of 4, which gives the value 3.04. This was an error in my own pre-registration.

**Exploratory, post hoc (not a verdict).** After the pre-registered value was observed, the same comparison was recomputed with block sums (energy-conserving). The relative L2 is 2.85·10⁻², which would be below the threshold. This value was computed after seeing the result and is reported only as exploratory. The pre-registered verdict for T3 remains **fail**.

**Consequences.**
1. CONTRATO-003 uses N = 128. A robustness rerun of its final configuration at N = 256 is required before the D2NN numbers are cited (gap G02). It will be labelled post hoc.
2. The pre-registered T3 cannot be rerun with corrected wording without starting a new contract. This contract remains a record of what was measured.
