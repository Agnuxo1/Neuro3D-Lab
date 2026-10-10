# CONTRATO-006b (POST HOC diagnostic, not a verdict)

**Status:** post hoc. Run after the CONTRATO-006 verdict was known. It does not replace or change that verdict (V2 remains **fail**).

**Question.** Is the CONTRATO-006 V2 failure caused by the quadrature of the Rayleigh–Sommerfeld reference at source pitch dx/2?

**Method.** Same tapered tilted wave I2 (sin θ = 0.498, super-Gaussian taper, ρ₀ = 18 mm), same grid N = 128, dx = 0.4 mm, λ = 0.75 mm, d = 18 mm. Three fields: (a) angular spectrum; (b) RS reference at source pitch dx/2 (the CONTRATO-006 reference); (c) RS reference at source pitch dx/4. Script: `experiments/diagnose_rs_quadrature.py`. Output: `experiments/results/diagnose_rs_quadrature.json`.

**Result (relative L2 error of the complex field).**

| Pair | Tapered I2 | Untapered I2 |
|---|---|---|
| (a) ASM against (b) RS dx/2 | 1.335 × 10⁻¹ | 5.219 × 10⁻¹ |
| (a) ASM against (c) RS dx/4 | 1.335 × 10⁻¹ | 5.218 × 10⁻¹ |
| (b) RS dx/2 against (c) RS dx/4 | **8.0 × 10⁻¹¹** | 2.4 × 10⁻³ |

Runtime 734.5 s; exit code 0. Recomputed V2 matches the registered value (1.33 × 10⁻¹).

**Reading.**
1. The reference quadrature is **not** the cause. RS at dx/2 and dx/4 agree to 8 × 10⁻¹¹ for the tapered input.
2. The discrepancy is between the two methods. The angular spectrum and the Rayleigh–Sommerfeld reference differ by 13 % for this smooth, tapered tilted wave at dx/λ = 0.53, and refining the reference does not change it.
3. The untapered input has a large difference because the finite reference window truncates the tilted wave, as already recorded in CONTRATO-006.
4. Not yet established: whether the error lies in the reference kernel implementation, in the periodic (FFT) boundary of the angular spectrum, or in the scalar model itself. A planned test is an analytic tilted plane wave on a large disc, checked against the exact plane-wave solution for each method. It is not run here.

**Consequence.** The simulator's propagator is not yet validated against an independent reference in the D2NN sampling regime. The CONTRATO-004 T1 pass (analytic plane waves on the grid) does not establish agreement with a reference method. Results from the D2NN experiments carry this as an open systematic uncertainty.
