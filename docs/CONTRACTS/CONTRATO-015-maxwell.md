# CONTRATO-015 — Two-dimensional Maxwell check of the scalar thin-element model (pre-registered)

**Status:** pre-registered on 2026-10-13, before any run of `scripts/completion_tools/maxwell_validation.py`. The guide's own probe (`neuro3d_completion.maxwell_probe`, 2D Ez/Hz at several resolutions, PML, time and angle variants, and one 3D run) is executed alongside as a descriptive record; it has no verdict.

**Question (guide phase F).** For the element that the scalar model represents by a pure phase delay, how far is a full-wave (Maxwell) solution from the thin-element prediction, at the project's scale (pixel 0.533 λ, glass n = 1.5, thicknesses 0.5 λ and 1 λ)?

## Setup

Meep 1.35.0 (conda-forge `pymeep`, Python 3.12, WSL Ubuntu 22.04; environment exported to `environments/meep-environment.yml`). Units of the vacuum wavelength; propagation along +x; 2D; PML 1 λ; cell 10 λ × 8 λ; Gaussian pulse source at x = −3 (centre frequency 1, width 0.15), transverse Gaussian of half-width 2 λ; DFT at the centre frequency; 200 time units after the source. Monitors at x = −1.5 (incident) and x = 2 (output), |y| ≤ 3. Runs: air, a uniform slab (thickness 1 λ), and the two-pixel element of the guide's probe. Polarisations Ez (TE, exactly scalar in 2D) and Hz (TM). Resolutions 24, 48 and 96 pixels per λ for Ez; 24 and 48 for Hz.

## Registered checks (thresholds from the guide's proposal, fixed here)

- **V0 vacuum:** scalar angular-spectrum propagation of the Meep air field from the incident to the output monitor agrees with the Meep air field at the output within 1e-2 (relative L2, |y| ≤ 1.5), at the finest resolution.
- **V1 slab:** the Meep slab/air transmission at the beam centre agrees with the normal-incidence Fabry–Perot formula within 0.02 (absolute complex difference), at the finest resolution.
- **V2 discretisation:** the element field at the output (divided by the air-field norm) changes by at most 1 % between the two finest resolutions.
- **V3 thin element:** at the finest resolution, for detector bins of one pixel width (|y| ≤ 1.6) holding at least 5 % of the peak bin power, the scalar thin-element prediction differs from Maxwell by at most 5 % in bin power, and by at most 0.05 rad in phase where the scalar amplitude is at least 30 % of its maximum. Evaluated twice: pure phase delay (the D2NN model) and phase delay with single-pass Fresnel amplitude 4n/(1 + n)². No global phase is removed.

If V0–V2 pass and V3 fails, the conclusion is that the scalar thin-element model, at this pixel scale and these thicknesses, is not accurate to the stated tolerances, and the D2NN results remain statements about the scalar model only. If V0, V1 or V2 fail, V3 is not interpreted.

## Scope

A 2D, two-pixel, normal-incidence element of one material. It is not the 3D D2NN, not a fabricated element and not a measurement. The guide's 3D probe is one resolution and is descriptive only.

## Registration

The commit that adds this file, before any run of the validation script.

## Results

To be filled after the run, without changing the sections above.
