# Physical model

This document fixes the conventions and equations used by `optlab`. All code must follow it. If an equation changes, it is changed here first, then in the code, under a new contract.

## 1. Conventions

- Complex scalar field `u(x, y)` in a transverse plane. Time dependence `exp(-i·ω·t)`.
- SI units: metres and seconds. The Blender scene is converted to metres with `scene.unit_settings.scale_length` (default: 1 Blender unit = 1 m).
- Square grid `n × n` with pitch `pixel_m`. Array indices are `[row, column] = [y, x]`.
- Medium between the bench planes: `medium_index` (default 1.0). Propagation uses `λ/medium_index`. The element phase uses the vacuum wavelength `λ` with `(n_mat − n_med)`.
- Wavenumber in a medium of index `n`: `k = 2π·n/λ`.
- Optical path phase: `φ = 2π·OPL/λ`, with `OPL = Σ n_i·d_i` (sum over the user-supplied segments; see §7).

## 2. Free-space propagation (angular spectrum)

For a distance `d` in a medium of index 1. In a medium of index `n`, replace `λ` by `λ/n`.

```
U = fft2(u)
H(fx, fy) = exp(i·2π·d·sqrt(1/λ² − fx² − fy²))   if fx² + fy² < 1/λ²
H(fx, fy) = 0                                     otherwise (evanescent)
P_d(u) = ifft2(H · U)
```

`fx`, `fy` are the frequencies of `numpy.fft.fftfreq(n, d=pixel_m)`.

- The continuous kernel is exact for propagating components. No paraxial approximation is made.
- Evanescent components are discarded. The field loses energy if it has spatial-frequency components above `1/λ`. The energy check (§8) does not measure this loss.
- Adjoint with respect to the complex inner product: `P_d^H(g) = ifft2(conj(H) · fft2(g))`.

The discrete implementation (`src/optlab/propagation.py`, `src/optlab/bench.py`) has two further limits. Let `dx = pixel_m` and `N = n`.

- **Sampling (Nyquist) limit.** Along each axis the grid represents spatial frequencies only up to `1/(2·dx)`. In the D2NN sampling regime of CONTRATO-004 (`dx/λ = 0.53`) this limit lies below the edge of the propagating band, `1/λ`. The highest propagating angles are therefore not sampled on that grid. CONTRATO-004 reports this limit; it does not test it.
- **Validity limit.** The discrete propagation is reliable only while the free-space distance satisfies `d < N·dx²/λ` (in a medium of index `n`: `d < N·dx²·n/λ`). Beyond this distance, aliasing of the transfer function invalidates the result. The simulator refuses any free-space segment that does not satisfy this condition, unless `allow_out_of_domain=True` is passed. A result obtained that way lies outside the validity domain and must be reported as such.

## 3. Phase elements and absorption

A plane of height `h(x, y)` (metres) with complex index `n_mat + iκ` in a medium of index `n_med`:

```
t(x, y) = exp(i·2π·(n_mat − n_med)·h/λ) · exp(−2π·κ·h/λ)
```

- In the bench format (`optlab.bench/1`), `h = height_npy × height_max_m`.
- `|t| = 1` if `κ = 0`.
- Intensity transmission through the plane: `|t|² = exp(−4π·κ·h/λ)`. The intensity absorption coefficient is `α = 4πκ/λ`, so `I ∝ exp(−α·h)`.
- Pure phase layers: `t = exp(iφ_k)` with `φ_k` free in `[0, 2π)`. For the D2NN, `φ_k` is either obtained from the height or optimised directly.

## 4. Interfaces: Fresnel

Normal incidence between `n1` and `n2`, without absorption:

```
r = (n1 − n2)/(n1 + n2)      R = r²      T = 1 − R
```

Oblique incidence, with Snell's law `n1·sinθi = n2·sinθt`:

```
r_s = (n1 cosθi − n2 cosθt)/(n1 cosθi + n2 cosθt)
r_p = (n1 cosθt − n2 cosθi)/(n1 cosθt + n2 cosθi)
t_s = 2 n1 cosθi/(n1 cosθi + n2 cosθt)
t_p = 2 n1 cosθi/(n1 cosθt + n2 cosθi)      (amplitude of E)
R = |r|²
T = (n2 cosθt)/(n1 cosθi) · |t|²
```

Sign convention for `r_p` (the one used in `fresnel.py`): `r_p` is defined with the electric field, so that `r_p = r_s` at normal incidence (`θi = 0` gives `(n1 − n2)/(n1 + n2)` for both). Some texts use the opposite sign, which gives `r_p = −r_s` at normal incidence. `R` and `T` do not depend on this sign; only the phase of `r_p` does. At Brewster's angle `r_p = 0`. For `θi` above Brewster's angle, with `n2 > n1`, `r_p > 0`.

Total internal reflection when `n1·sinθi > n2`: `R = 1`, `T = 0`.

Reflections between layers are not included in version 0.1 (see §8). The Fresnel relations are tested in CONTRATO-001.

## 5. Interferometry

50/50 beam splitter `B = (1/√2)·[[1, i], [i, 1]]`. Phase shifter `P(θ) = diag(e^{iθ}, 1)`.

```
MZI(θ, φ) = B · P(θ) · B · P(φ)
```

Properties tested in CONTRATO-001: unitarity of `MZI(θ, φ)`, checked on 50 random pairs `(θ, φ)`; and, with `φ = 0`, `|T11|² = sin²(θ/2)`.

## 6. Diffractive neural network (D2NN)

Input `u0`. `L` phase layers and `L+1` propagations:

```
b_k = P_{d_k}(a_{k−1}),   a_0 = u0,   a_k = b_k · exp(iφ_k)     (k = 1..L)
f   = P_{d_{L+1}}(a_L)
I   = |f|²
s_c = scale · Σ_{x ∈ region_c} I(x)
p   = softmax(s),   L_ce = −log p_y
```

Analytic gradient, with the convention `ḡ = ∂L/∂z*`:

```
ḡ_f       = (∂L/∂I) · f,      ∂L/∂I = scale · Σ_c (p_c − δ_{cy}) · mask_c
ḡ_{a_L}   = P^H_{d_{L+1}}(ḡ_f)
for k = L..1:
    ∂L/∂φ_k = −2 · Im(conj(ḡ_{a_k}) · a_k)
    ḡ_{b_k} = ḡ_{a_k} · conj(exp(iφ_k))
    ḡ_{a_{k−1}} = P^H_{d_k}(ḡ_{b_k})
```

The gradient is checked against central finite differences in CONTRATO-001.

**Sampling regime.** The D2NN contracts use `dx/λ = 0.53`. The sampling limit of §2 applies.

**Periodic-boundary effect of the N = 128 grid (about 4 %).** The D2NN grid `N = 128` is affected by periodic wrap-around. CONTRATO-004, test T2, compares the intensity after 18 mm computed on `N = 128` and on `N = 256` (same `dx`). The relative L2 difference is `4.4·10⁻²`, against a threshold of `2·10⁻²`, so the test fails. The periodic wrap-around changes the intensity in the `N = 128` grid by about 4 percent. CONTRATO-003 uses `N = 128`. Its D2NN results must be checked on a larger grid before they are cited. This is an open item.

**Status of CONTRATO-003.** The digit-classification D2NN (3 phase layers, 18 mm spacing, `N = 128`) is in progress. No test accuracy is reported in this document. Validation (DEV) accuracies are preliminary and are not results.

## 7. Optical paths and Blender

Blender does not simulate waves. It is used for geometry: the positions of the planes and the medium index of each material.

The optical path length from ray-traced path lengths is **not implemented**. `trace_optical_path` (in `blender/neuro3d_lab/bench_io.py`) casts a ray and returns the successive hits. Each hit gives the object name, the distance along the ray in metres, and the medium index stored on the object (`n_medium`). It does not return an optical path length.

The optical path length is computed only from segments supplied by the user, as `OPL = Σ n_i·d_i` (`optlab.units.optical_path_length`). For a ray that crosses planes at `z_1 < z_2 < ...`, the user may take `d_i = z_{i+1} − z_i` and `n_i` for the medium between those planes. The add-on's self-test checks that the hit distances match the plane positions of the demo bench, including at scale_length values 1, 0.01 and 0.001.

For visibility only, the bench planes are enlarged 15 times in x and y in the Blender scene. Their z spacing is physical. The Blender bench demo uses `N = 256`, `dx = 8 µm`, `λ = 633 nm`, and planes at 0, 20, 40 and 60 mm.

## 8. Explicit limitations (version 0.1)

- Scalar model: no polarisation and no vector fields. For oblique incidence on phase planes, the scalar approximation is not checked and should not be relied on.
- No multiple reflections between layers. Propagation is forward only.
- No spectral dispersion: `n` is constant in `λ`.
- No calibration in watts: intensity is relative and normalised.
- Plain angular-spectrum propagation is valid only for `d < N·dx²/λ` (§2). The simulator enforces this unless it is explicitly overridden.
- Periodic boundary: the D2NN grid `N = 128` has an effect of about 4 percent (§6, CONTRATO-004 T2).
- Energy check: CONTRATO-002 checks conservation of energy for band-limited fields with no losses. It does not measure losses into evanescent components, which the propagator discards.
- Detector regions are rectangular boxes. Non-overlap is a convention; the code does not check it.
- No physical hardware experiment exists. All results are simulations. No figure in this repository implies a fabricated device.

## 9. Validation contracts

Each numerical check has a contract written before the test is run, with thresholds fixed in advance. A threshold is not relaxed after the result is seen. Amendments made before execution are dated and recorded in the contract. A failure is published with its numerical value. See [CONTRACTS/](CONTRACTS/).
