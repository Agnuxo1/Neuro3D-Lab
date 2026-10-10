# Related work

**Status.** Every entry below was opened and re-checked on 2026-10-10 (fetch by one agent, independent re-check by a second). Descriptions are paraphrased. No numbers are taken from a source that could not be opened. Entries marked **not verified** are listed for follow-up only.

## Diffractive optical neural networks

**Lin, X. et al. (2018). All-optical machine learning using diffractive deep neural networks.** *Science* 361(6406), 1004–1008. DOI [10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084); arXiv [1804.08711](https://arxiv.org/abs/1804.08711).
Authors in the journal record (Crossref): Xing Lin, Yair Rivenson, Nezih T. Yardimci, Muhammed Veli, Yi Luo, Mona Jarrahi, Aydogan Ozcan. The arXiv abstract page lists six authors and omits Yi Luo; the journal record is used here.
*What it does:* passive diffractive layers, designed by deep learning, jointly perform a classification or imaging function. The authors built 3D-printed versions that classified handwritten digits at terahertz frequencies.
*Relevance:* the direct precedent for the phase-only D2NN simulated in this repository.

**Mengu, D., Luo, Y., Rivenson, Y., Ozcan, A. (2020). Analysis of diffractive optical neural networks and their integration with electronic neural networks.** *IEEE Journal of Selected Topics in Quantum Electronics* 26(1), article 3700114. DOI [10.1109/JSTQE.2019.2921376](https://doi.org/10.1109/JSTQE.2019.2921376); arXiv [1810.01916](https://arxiv.org/abs/1810.01916).
Year: the issue is dated 2020 (January–February); the electronic version is dated 2019 (June). Both dates are on record and both are given here.
*What it does:* analyses the training of diffractive networks, including the loss function and vanishing gradients in back-propagation, and studies a phase-only diffractive front end combined with an electronic fully connected layer.
*Relevance:* the hybrid optical–electronic design, and the analysis of the loss function and gradients, which bears on the training choices in CONTRATO-003 to 007.

**NDNN (THUMetaOptics). Scalable multilayer diffractive neural network with all-optical nonlinear activation.** arXiv [2504.13518](https://arxiv.org/abs/2504.13518) (2025); code at [THUMetaOptics/NDNN](https://github.com/THUMetaOptics/NDNN).
*What it does:* a folded multilayer diffractive network that uses one spatial light modulator and the optical nonlinearity of a mirror-coated silicon wafer as its activation. Training uses back-propagation with a mean-squared-error loss over ring-shaped detectors.
*Licence:* **none found** in the repository or README. The code must not be copied into this project; only the paper is cited.
*Relevance:* multilayer scaling with a nonlinear activation, a contrast to the linear phase-only model used here.

## Simulation software

**TorchOptics.** Repository [MatthewFilipovich/torchoptics](https://github.com/MatthewFilipovich/torchoptics); arXiv [2411.18591](https://arxiv.org/abs/2411.18591) (2024), *TorchOptics: An open-source Python library for differentiable Fourier optics simulations*.
*What it does:* a PyTorch library for GPU-accelerated, differentiable Fourier optics, including polarised and partially coherent fields.
*Relevance:* a natural target for the planned independent PyTorch re-implementation (CONTRATO-009). Outreach issue [#99](https://github.com/MatthewFilipovich/torchoptics/issues/99) is open with no reply yet.

**onn-simulation.** Repository [feng-spinning/onn-simulation](https://github.com/feng-spinning/onn-simulation).
*What it does:* a PyTorch simulation of a diffractive deep neural network on MNIST digits, with variants that include nonlinear activation, incoherent propagation and changes of the propagation distance.
*Relevance:* a comparable D2NN digits protocol. Outreach issue [#10](https://github.com/feng-spinning/onn-simulation/issues/10) is open with no reply yet. Its licence was not recorded in this pass.

**POPPY (Physical Optics Propagation in Python).** Repository [spacetelescope/poppy](https://github.com/spacetelescope/poppy); documentation at poppy-optics.readthedocs.io. Primary reference: Proc. SPIE 8442 (2012); ASCL 1602.018.
*What it does:* Fraunhofer and Fresnel propagation through optical systems, developed for the James Webb Space Telescope.
*Relevance:* an established scalar propagation code and a cross-check candidate for the propagation method used here. The 2012 paper's full author list was not read; cite the repository and the ASCL entry.

**Blender Optics Simulator.** Repository [emircbngl/blender-optics-simulator](https://github.com/emircbngl/blender-optics-simulator); licence GPL-3.0-or-later (README).
*What it does:* a Blender add-on that lays out optical benches in 3D and traces rays and Gaussian beams through lasers, mirrors, lenses and detectors. It exposes the bench state through a local MCP bridge.
*Relevance:* a ray and Gaussian-beam tool, not a scalar diffraction solver. It is complementary to Neuro3D. Outreach issue [#86](https://github.com/emircbngl/blender-optics-simulator/issues/86) is open with no reply yet. The Zenodo DOI was not confirmed and is not cited.

**BlenderPhotonics.** Repository [NeuroJSON/BlenderPhotonics](https://github.com/NeuroJSON/BlenderPhotonics); DOI [10.1117/1.JBO.27.8.083014](https://doi.org/10.1117/1.JBO.27.8.083014) (*J. Biomedical Optics*, 2022).
*What it does:* a Blender add-on that converts objects or NIfTI volumes to tetrahedral meshes and runs mesh-based and voxel-based Monte Carlo light simulations for tissue optics.
*Relevance:* a Blender photonics tool for tissue optics, not for coherent diffraction. Outreach issue [#16](https://github.com/NeuroJSON/BlenderPhotonics/issues/16) is open with no reply yet. The journal page could not be read; the DOI was checked only as a redirect.

## Not verified (listed for follow-up, not cited with numbers)

**Hu, Q. et al. (2025). Optical diffraction neural network in-situ clustering for full self-learning ability.** OECC/PSC 2025, session TuG3-07 (conference presentation).
The programme page returned HTTP 403 to the verification tool, so the abstract was not read. The first author and author list come from a search snippet. No DOI, arXiv id or journal version was found. The accuracy quoted in the snippet is **not verified** and is not used anywhere in this project.
*Follow-up:* obtain the full paper, then verify it before citing it.
