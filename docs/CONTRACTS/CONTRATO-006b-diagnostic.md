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

## Addendum (post hoc, same day): where the V2 discrepancy sits

Two further checks, both post hoc and not part of the verdict.

- **Central nodes, large-window reference.** The reference was recomputed on a 205 mm window at source pitch dx/2 (1024 × 1024 sources). The angular spectrum at N = 128 agrees with it to **4 × 10⁻⁷** on the central 9 × 9 grid nodes (|x| ≤ 3.2 mm), with observation points on grid nodes (no interpolation).
- **Intermediate region.** On the central 33 × 33 nodes (|x| ≤ 6.4 mm, where |field| is between 0.39 and 1.0) the agreement is **2 × 10⁻⁶**.

**Hypothesis (not verified over the full grid).** The V2 relative error of 0.133 is computed over all 128 × 128 nodes, including the outer region up to about ±25 mm where the taper is still substantial. The V2 reference used a 51.2 mm source window, which truncates the tilted wave near its edges. The finite-window reference differs from the large-window one there, and that difference dominates the full-grid relative error. A direct test needs the 205 mm reference on all 128 × 128 nodes, which is computationally heavy and is left open.

**Consequence for the overall reading.** The angular spectrum agrees with an independent Rayleigh–Sommerfeld reference to a few parts in 10⁶ in the region where the field is strong. The CONTRATO-006 V2 failure is therefore best attributed to the finite-window reference and the full-grid metric, not to the central propagation. This explanation remains a hypothesis until the full-grid test is run. CONTRATO-006 V2 remains **fail** as registered.
