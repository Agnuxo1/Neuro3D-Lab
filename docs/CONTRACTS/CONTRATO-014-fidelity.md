# CONTRATO-014 — Discretisation fidelity of trained D2NN masks (pre-registered, TRAIN/DEV only)

**Status:** pre-registered on 2026-10-13, **before** any run of `scripts/completion_tools/fidelity_study.py`. The commit that adds this file and its push are the registration. TEST and QMNIST50K are never read: the tool refuses reserved splits by name and by recorded hash.

**Question (guide §14.2, Q1).** Simulations of diffractive networks commonly represent each physical pixel by one numerical sample. Here the pixel is 0.4 mm = 0.533 λ. Do phase masks trained with one sample per pixel (refinement r = 1) keep their accuracy when the *same* physical pixels are simulated as piecewise-constant elements with tied subpixels (r = 2, 4, 8, 16) and with direct Rayleigh–Sommerfeld summation? And what accuracy do masks reach when they are trained in a better-resolved model?

**Prior evidence that motivated the question (not a result of this contract).** With the historical seed-0 masks and smooth inputs, r = 1 → 2 changes detector scores by 270 %, r = 4 → 8 by 6 %, with observed order 2.2 (CONTRATO-012). On a 16-pixel random-phase three-layer network, the r = 1 model differs from direct RS by 44–52 % in scores (`evidence/rs-network-n16.json`). Those masks were trained on 8 × 8 digits, so this does not yet say anything about accuracy.

## Fixed design

- Data: writer-disjoint MNIST (`data/mnist-writer`). Training on the first 10,000 TRAIN images in file order; evaluation on the full DEV split (10,095 images, 91 writers, disjoint from TRAIN), except r = 16, evaluated on the first 2,000 DEV images because of its cost.
- Optics: n = 128, pixel 0.4 mm, λ = 0.75 mm, 4 × 18 mm, three phase layers, aperture exterior, no filter, complex128. Training discretisations (r, p): (1, 2), (1, 4), (2, 2), (4, 2), (8, 2). Evaluation discretisations: (1, 2), (1, 4), (2, 2), (4, 2), (8, 2), (16, 2) for every trained mask set.
- Training: fixed, not tuned: β = 100, Adam lr = 0.02, batch 16, 10 epochs, last epoch, seeds 0, 1, 2 (`benchmark.train`, same initialisation and batch order rules as CONTRATO-013). The per-epoch monitor uses the first 512 DEV images and selects nothing.
- Refinement keeps 49,152 physical phases; subpixels share the phase of their pixel; nothing is retrained when the evaluation discretisation changes.
- r = 16 uses a constructor that builds the same buffers directly on the GPU; its identity with the standard constructor is checked at r = 8 on 32 DEV images (tolerance 1e-10 relative), and the tool refuses any m ≥ 4096 evaluation until that recorded check has passed. At m ≥ 4096 the evaluation batch is 4 (memory only; scores do not depend on the batch).
- β is read from each run's `architecture.json`, so the same tool can re-discretise the seed-0 phases of CONTRATO-013 (its refinement line).
- Direct RS (seed 0 of every training discretisation): DEV images 0–3, quadrature q = 1 and 3 subcells per pixel side, absorbing exterior, `rs_network.rs_network_fields`.

## Registered verdicts

- **F1 (fidelity gap).** For masks trained at r = 1 (both p = 2 and p = 4), the mean DEV accuracy over seeds 0–2 evaluated at r = 8, p = 2 is lower than the accuracy evaluated in the training discretisation by more than 0.05 (absolute). F1 holds only if it holds for both paddings; it is undecided unless both paddings have all three seeds. Because r = 8 is itself not converged, the same drop against r = 16 (on its 2,000-image subset) is reported as a robustness line.
- **F2 (approximate convergence of training at r = 8).** For masks trained at r = 8, p = 2, the mean accuracy on the first 2,000 DEV images differs by at most 0.01 between evaluation at r = 8 and r = 16; undecided unless all three seeds are available.
- Descriptive, without a threshold: the full train × evaluation accuracy matrix; the accuracy of masks trained at r = 8 against those trained at r = 1 (each evaluated at r = 8), paired by seed; for seed 0 of each training discretisation, RS q = 1 versus q = 3 score differences and the agreement between the core evaluated **in its training discretisation** and RS. The RS lines rest on four images and on q = 3, which is itself unconverged (q 3 → 9 changes scores 3.4 % on the 16-pixel network), so they are anecdotal.

Thresholds: 0.05 is a large practical effect (more than ten times the Wilson half-width at n ≈ 10,000). F2 is a paired difference on the same 2,000 images: its sampling error comes only from discordant images, about √(p_d/2000) ≈ 0.003 for a discordance p_d of 2 %, so 0.01 is about three standard errors; writer clustering on the DEV makes this optimistic.

## Scope and limits

- Scalar thin-element model; DEV only; one fixed setting; three seeds; reduced training budget (10,000 images, 10 epochs), so absolute accuracies are below those of CONTRATO-013 and are not benchmark results.
- p = 2 is used at r ≥ 2 for cost; at r = 1 the window error between p = 2 and p = 4 is about 1.3 % in scores (CONTRATO-012) and both are trained.
- r = 16 is itself not proved converged; F2 measures the step 8 → 16 only.
- Generalising beyond this geometry requires the prior-art review of the sampling used in published diffractive-network simulations (guide §23.4).

## Outputs

`completion_results/fidelity/train_r*_p*_seed*/` (phases, histories, per-sample DEV scores, RS records) and `evidence/fidelity-summary.json`.

## Registration

- Pre-registration commit: `9521a9bfd30a400479015ff3b70ba9454a314131` (2026-10-13), pushed to `origin/completion/2026-10-12` before any run of this contract; hash manifests in `evidence/protocol-manifest-*.json` (commit b7f8202).
- External registry (OSF or Zenodo): pending; it needs the owner's account.

## Results

To be filled after the runs, without changing the sections above.
