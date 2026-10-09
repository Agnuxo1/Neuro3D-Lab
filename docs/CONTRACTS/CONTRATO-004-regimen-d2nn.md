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

- Same physical digit (25.6 mm wide). Compare dx = 0.4 mm (N = 128) against dx = 0.2 mm (N = 256, same physical window, digit upsampled to the same physical size).
- The dx = 0.2 mm intensity is block-averaged 2×2 to the dx = 0.4 mm grid before comparing.
- Metric: relative L2 difference of intensity.
- **Threshold:** ≤ 0.05.

## What is not claimed by this contract

- Agreement with an independent reference solver (Rayleigh–Sommerfeld direct summation, FDTD): not tested here. Remains an open gap (G01, P0).
- Vector effects, polarization, or any statement about fabricated devices.
- Results beyond the single wavelength and the single geometry above.

## Result registry

Filled in after running `experiments/run_regime_d2nn.py`. Each entry: local date, commit, measured values, verdict per test.
