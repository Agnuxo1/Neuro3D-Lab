# Prior-art review for the discretisation-fidelity question (guide §23.4)

**Date:** 2026-10-13. **Reviewer:** the completion agent (Claude), for the project owner. **Purpose:** decide what this project may and may not claim about sampling one numerical point per sub-wavelength pixel (CONTRATO-012, CONTRATO-014) before any result of CONTRATO-014 exists. This review is a registered record, not a systematic review: it used one general web search service and arXiv full texts, not Scopus or Web of Science (no access).

## Queries (exact text, all on 2026-10-13)

1. `diffractive deep neural network simulation sampling per neuron sub-wavelength pixel discretization accuracy fabricated`
2. `diffractive neural network "oversampling" OR "upsampling factor" neuron size simulation grid D2NN`
3. `Lin 2018 all-optical machine learning diffractive deep neural network supplementary sampling 0.4 mm neuron simulation pixel`
4. `diffractive optical network numerical sampling convergence "subpixel" OR "sub-pixel" piecewise constant neuron simulation accuracy mismatch experiment`
5. `arXiv 2606.07896 Beyond the Thin-Layer Limit differentiable volumetric training visible-range diffractive neural networks`

Inclusion: work that states how a diffractive network's fields are sampled relative to its feature (neuron) size, or that measures a simulation–experiment or model–model gap attributable to the numerical model. Full texts read: arXiv 1810.01916 and 1909.06553 (text extracted from the arXiv PDFs).

## Findings

| Work | What it states (paraphrased; short quotes only) | Relevance |
|---|---|---|
| Mengu, Luo, Rivenson, Ozcan, *Analysis of diffractive optical neural networks and their integration with electronic neural networks*, IEEE JSTQE 26 (2020), arXiv 1810.01916 | Fields "were sampled on a grid with a spacing of ~0.53λ"; each layer has neurons of 0.53λ × 0.53λ; angular spectrum propagation. | One numerical sample per 0.53 λ neuron, i.e. refinement 1 in this project's terms. Its MNIST/Fashion-MNIST accuracies are therefore results of the sampled model. |
| Luo, Mengu, Yardimci, Rivenson, Veli, Jarrahi, Ozcan, *Design of task-specific optical systems using broadband diffractive neural networks*, Light Sci. Appl. 8, 112 (2019), arXiv 1909.06553 | 0.5 mm printable pixel; "the simulation window was oversampled by 4 times"; each feature represented on a 4 × 4 grid whose 16 elements share one thickness. | Tied-subpixel oversampling (refinement 4) is established practice in this group's broadband designs. **This project's tied-subpixel construction is not new.** |
| Pyramid diffractive networks, arXiv 2308.15019 | Feature 0.5 mm (~0.67 λ), field sampling 0.25 mm (~0.33 λ) "to ensure precise modeling". | Refinement 2 used for experimental verification at THz. |
| Lin et al., *All-optical machine learning using diffractive deep neural networks*, Science 361, 1004 (2018) | 0.4 THz, 8 × 8 cm layers; the neuron size 0.4 mm is consistent with later papers of the group. | The simulation sampling of the original paper was **not verified** here (supplement not read). |
| Jayakody, Wadduwage, *Beyond the thin-layer limit: differentiable volumetric training for visible-range diffractive neural networks*, arXiv 2606.07896 (2026) | Thin-layer models break down for thick, densely coupled layers; volumetric training compared with thin-layer training under FDTD. | Model error beyond the thin-element approximation (the vector question of guide phase F), not the sampling question. |
| Sun et al., Int. J. Optics (2021), 10.6 µm D2NN | Accuracy depends on neuron grid and layer spacing in simulation. | Design-space study; not a discretisation-convergence test. |
| Volume-holographic D2NN, Sci. Rep. (2024) | 88.7 % simulated vs 18 % measured before aberration-aware training. | Shows large simulation–device gaps exist; attributed to fabrication, not sampling. |

## Decisions for this project

1. **No novelty claim for tied-subpixel refinement** (prior use: Luo et al. 2019, refinement 4; arXiv 2308.15019, refinement 2).
2. **Permitted framing for CONTRATO-014:** a quantitative measurement, for this geometry, of how masks trained in the one-sample-per-0.53 λ model (the sampling of Mengu et al. 2020) behave when the same pixels are simulated piecewise-constant at finer quadrature and with direct Rayleigh–Sommerfeld summation. Whether this has been measured before for trained masks was **not established** by this search; any manuscript must say "to our knowledge, after the search recorded in `docs/LITERATURE_REVIEW.md`" and cite the works above.
3. **Not claimed:** that published experimental results are wrong. Experimental devices may agree with experiments for other reasons (fabrication resolution, detector integration, the group's own oversampling in later work).
4. To be done before submission: read the Lin et al. 2018 supplement and the 2019–2025 Ozcan-group methods sections for their sampling factor; search Optica Publishing Group and IEEE Xplore directly.

## Sources

- https://arxiv.org/abs/1810.01916
- https://arxiv.org/abs/1909.06553 and https://www.nature.com/articles/s41377-019-0223-1
- https://arxiv.org/abs/2308.15019
- https://arxiv.org/abs/1804.08711 (Lin et al. preprint); DOI 10.1126/science.aat8084
- https://arxiv.org/abs/2606.07896
- https://onlinelibrary.wiley.com/doi/10.1155/2021/6667495
- https://www.nature.com/articles/s41598-024-82791-z
