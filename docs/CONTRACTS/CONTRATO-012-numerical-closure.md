# CONTRATO-012 — Numerical closure record: compatibility, window, quadrature and scalar references

**Status:** verification record (phases A, B, B2 and C of the completion guide). The criteria were proposed in the completion guide v1 (2026-10-12) before any of these runs; the guide v2 review had already reported most of the numbers, so this is a **reproduction against fixed criteria, not a blind confirmatory test**. Every number below was regenerated on 2026-10-12/13 on this workstation (Windows 10, Python 3.13.7, NumPy 2.2.6, SciPy 1.15.1, PyTorch 2.6.0+cu124; lock in `environments/`). Nothing in CONTRATO-001 to CONTRATO-011 is changed.

## A. Reproduction and compatibility

| Check | Result | Evidence |
|---|---|---|
| Test suite (33 historic + 34 guide v1 + 22 guide v2) | 89 passed | `pytest -q` |
| New implementation vs legacy `optlab` (periodic, p = 1, r = 1), four complex inputs | field 1.69e-15, intensity 2.98e-15, scores 2.88e-15, loss 4.4e-16, gradient 3.16e-15: **PASS** | `evidence/legacy-compatibility.json` |
| NumPy vs Torch equivalence gate on CUDA (deterministic mode) | scores 5.9e-16, loss 0, gradient 1.3e-15: **PASS** | `evidence/gpu-probe-rtx3090.json` |
| Historic scripts replayed at e5ebb98 in a detached checkout (CONTRATO-002 validation, negative control, CONTRATO-010, CONTRATO-011) | every numeric field identical except wall-clock `elapsed_s` | `evidence/phaseA-historical-replay.json` |
| Synthetic end-to-end smoke (9 models, 2 seeds) | PASS (software check only) | local log |

## B. Criteria (proposed in the guide before the runs) and outcomes

Relative L2 norms; no phase alignment, translation or renormalisation.

| Criterion | Threshold | Case | Value | Outcome |
|---|---|---|---|---|
| RS reference: quadrature refinement | ≤ 1e-3 | smooth Gaussian, n = 32 | 5.0e-15 | pass (trivial: smooth integrand, guide E5) |
| | | pixelated random screen, q 9 → 27 | 3.2e-4 | pass |
| | | full 3-layer random network, n = 16, q 3 → 9 | 3.4e-2 | **fail** |
| RS reference: source extension | ≤ 1e-3 | Gaussian n = 32 | 4.2e-16 | pass |
| Operator vs RS | ≤ 1e-2 | Gaussian n = 32 / n = 16 | 2.1e-8 / 5.5e-4 | pass |
| | | pixelated screen, r = 1, any p ≥ 16 | 4.8 %–8.3 % | **fail** (one sample per pixel) |
| | | pixelated screen, r = 3, p = 64 | 0.78 % | pass |
| | | random network n = 16, r = 3, p = 32 vs RS q = 9 | 3.45 % | **fail** (limited by the RS q = 9 quadrature itself) |
| Window p → 2p, historical masks, r = 1 (smooth inputs; 64 DEV images) | field ≤ 1e-2 | p 4 → 8 | 2.62 % ; 2.55 % | **fail** |
| | intensity ≤ 2e-2 | p 4 → 8 | 1.89 % ; 2.06 % | pass ; **fail** |
| | scores ≤ 1e-2 | p 4 → 8 | 0.12 % ; 0.15 % | pass |
| | labels ≤ 0.005 | p 4 → 8 | — ; 0 | pass |
| | all four | p 8 → 16 | ≤ 0.61 % ; ≤ 0.59 %, 0 labels | pass |
| Refinement r → 2r, historical masks, p = 2 (smooth; DEV) | scores ≤ 1e-2 | r 1 → 2 | 270 % ; 259 % | **fail** |
| | | r 2 → 4 | 27.9 % ; 26.9 % | **fail** |
| | | r 4 → 8 | 6.0 % ; 5.8 % | **fail** |

Evidence: `evidence/rs-gaussian-n16.json`, `rs-gaussian-n32.json`, `pixel-reference-random-seed0.json`, `rs-network-n16.json`, `boundary-historical-r1.json`, `boundary-historical-dev64-r1.json`, `pixel-historical-p2.json`, `pixel-historical-p4.json`, `pixel-historical-dev64-p2.json`, `v2-review/crosscheck.json`.

### Observed order and Grid Convergence Index (Celik et al. 2008, norm-based form)

| Series | Observed order (scores) | Estimated remaining error at the finest level | GCI |
|---|---|---|---|
| refinement 1-2-4, p = 2 | 3.27 | — | not asymptotic (r = 1 is another model) |
| refinement 2-4-8, p = 2 | 2.22 | 1.6 % | 2.1 % |
| window 2-4-8, r = 1 | 3.45 | — | field GCI 0.77 % |
| window 4-8-16, r = 1 | 2.16 | 0.008 % | field GCI 0.23 % |

`evidence/gci-pixel-historical-p2.json`, `evidence/gci-boundary-historical-r1.json`. The 2-4-8 refinement triplet is a single triplet; the stability of the order is not demonstrated (refinement 16 is pending, see below).

### Independent cross-check of the two implementations

n = 8, one layer, d = 6 mm, r = 3: the FFT core approaches direct RS (q = 3) from 9.3 % (p = 8) to 0.29 % (p = 64) and stalls; adding evanescent components to the core lowers it to 0.21 % (p = 64) and 0.045 % (p = 128). The stall is the deliberate omission of evanescent waves; the two independent implementations agree (`evidence/v2-review/crosscheck.json`).

## C. Error budget and certificates

The contraction budget, the score-error bound β(2‖f‖ε + ε²) and the margin certificate are implemented and unit-tested (`core.score_error_bound`, `core.certified_label`, `evidence.contraction_budget`). No operator-error bound η has been **proved** for the finite-window, piecewise-constant model: the values above are diagnostic differences between discretisations. Therefore **no label is certified**. Any margin analysis reported later uses measured differences and is labelled "estimate-based, not a certificate".

## Conclusions of this record

1. The new implementation reproduces the legacy model exactly where they overlap, and the NumPy and Torch implementations agree to rounding on CPU and CUDA.
2. At one numerical sample per 0.533 λ pixel (r = 1) the model is **not** a converged simulation of a piecewise-constant phase element: scores change by more than 250 % from r = 1 to r = 2, and r = 4 → 8 still changes them by 6 %. Historical results (CONTRATO-003 to CONTRATO-011) and the CONTRATO-013 benchmark describe the sampled model, not a fabricated pixelated device.
3. For the sampled model, padding 4 is adequate for scores and labels but not for fields; padding 8 meets every window criterion.
4. A pixelated-device simulation converged to about 2 % in scores needs r ≈ 8 (m = 2048 at p = 2), which costs 0.032 s per training sample on the RTX 3090 (`evidence/gpu-probe-rtx3090.json`); the consequences for trained masks are measured in CONTRATO-014.

## Pending

- Refinement 16 (m = 4096) for a second refinement triplet and order stability.
- Direct RS at full scale (n = 128, q = 3) on trained masks (CONTRATO-014).
- Vector (Maxwell) validation of the thin-element approximation (guide phase F).
