# CONTRATO-009 (pre-registered): independent PyTorch replication of the D2NN forward pass and gradient

**Status:** pre-registered on 2026-10-10, before any code for this contract was written. The thresholds below are fixed and are not changed after a result is seen. Failures stay in the record.

**Question.** Does an independent implementation in PyTorch (float64, CPU), written without reusing the optlab code, reproduce the optlab `DiffractiveNetwork` on one configuration: the detector intensities, the class scores, the loss, and the analytic gradient with respect to the phase masks?

**What this does and does not test.** It tests the implementation of the forward model and of the gradient. It uses the same physical model, namely the scalar angular-spectrum propagator with evanescent components removed. It does not test the physics against an independent reference; that is CONTRATO-006 and its diagnostics (006b, 006c). A difference in an evanescent convention would appear here as a failure, and it would be reported as a finding, not fixed by editing the tolerance.

## Configuration (fixed)

- Grid n = 32, pixel 0.4 mm, wavelength λ = 0.75 mm, distances d = [18, 18, 18] mm (L = 2 phase layers), scale = 1.0.
- Phase masks: optlab default, `np.random.default_rng(0).uniform(0, 2π, (L, n, n))`, as set by `DiffractiveNetwork(seed=0)`. The replication takes the phases from optlab's instance, so that both sides use identical phases.
- Detectors: K = 10 disjoint 4 × 4 square masks. Top-left corners: rows r ∈ {6, 22} and columns c ∈ {1, 7, 13, 19, 25}. Mask k sits at row r[k // 5], column c[k % 5].
- Inputs: 4 complex fields u₀ = A·exp(iθ) with A ~ Uniform(0.2, 1) and θ ~ Uniform(0, 2π) on the 32 × 32 grid, from `np.random.default_rng(1)`. Labels y = [0, 3, 5, 9].

## Independent implementation (requirements)

- Written from the definitions: transfer function H(fx, fy) = exp(i·d·k_z) with k_z = √(k² − kx² − ky²), k = 2π/λ, kx = 2π·fx, ky = 2π·fy, frequencies from `torch.fft.fftfreq(n, d=pixel)`. Components with k² − kx² − ky² ≤ 0 are set to zero (evanescent).
- Forward: a₀ = u₀; for k = 1..L: a_k = ifft2(H_{d_{k-1}} · fft2(a_{k-1})) · exp(i·φ_k); f = ifft2(H_{d_L} · fft2(a_L)); I = |f|²; s_c = Σ mask_c · I; loss = −log softmax(s)_y.
- Gradient: from autograd on the same loss, with respect to the phase tensor φ (real, float64).
- Imports from optlab: only `DiffractiveNetwork` for the reference outputs and the optlab phases. The replication does not call `angular_spectrum` or `loss_and_grad`.

## Pre-registered tolerances

| Test | Quantity | Pass if |
|---|---|---|
| T1 | intensities I, relative L2 over the grid, maximum over the 4 inputs | ≤ 1e-9 |
| T2 | class scores s, maximum absolute difference / max |s| | ≤ 1e-9 |
| T3 | loss, absolute difference, maximum over inputs | ≤ 1e-10 |
| T4 | gradient with respect to φ, optlab `loss_and_grad` against torch autograd, relative L2, maximum over inputs | ≤ 1e-8 |
| T5 | optlab gradient against central finite differences (step h = 1e-6) on 5 phase entries chosen with `default_rng(2)`, relative error with floor 1e-8 | ≤ 1e-5 |

**Verdict.** PASS if T1 to T5 all pass. FAIL otherwise, with each value reported. A FAIL is a result. It is not repaired by changing the tolerances or the configuration.

## Output

- Script: `experiments/replicate_d2nn_pytorch.py`. Output: `experiments/results/replicate_d2nn_pytorch.json`.
- Runs on CPU only, float64, no GPU queue use. Expected runtime: seconds.
- Run with `python -B` and `-p no:cacheprovider` where pytest is involved.
