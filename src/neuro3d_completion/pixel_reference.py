"""Independent Rayleigh-Sommerfeld reference for one pixelated (piecewise-constant) phase screen.

The historical model samples each physical phase pixel once (refinement r=1). A
fabricated element is piecewise constant. This module compares, at the physical
pixel centres of the output plane:

* RS(q): exact scalar RS1 kernel, midpoint quadrature with q x q subcells per pixel,
  aligned with pixel boundaries (independent of FFTs and periodic windows);
* ASM(r, p): the core propagator with refinement r and padding p.

Point-sample convention is used for both, so no normalisation is fitted. Odd
refinements put a numerical sample exactly at every physical pixel centre.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .core import Optics, DiffractiveNetwork, positive
from .evidence import record
from .validation import rayleigh_sommerfeld, relative


def _check_odd(values, name):
    values = tuple(int(v) for v in values)
    if not values or any(v < 1 or v % 2 == 0 for v in values) or sorted(set(values)) != list(values):
        raise ValueError(f"{name} must be increasing distinct odd integers")
    return values


def screen_function(phases, pixel_m, envelope_width_m):
    """Return f(x, y): Gaussian envelope times the pixel phase; zero outside the support.

    Rows increase downwards (+y up), matching detector_masks. Coordinates are metres.
    """
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 2 or phases.shape[0] != phases.shape[1] or not np.isfinite(phases).all():
        raise ValueError("phases must be a finite square array")
    n, a, w = phases.shape[0], positive(pixel_m, "pixel_m"), positive(envelope_width_m, "envelope_width_m")

    def field(x, y):
        col = np.floor(x/a + n/2).astype(int)
        row = np.floor(-y/a + n/2).astype(int)
        inside = (col >= 0) & (col < n) & (row >= 0) & (row < n)
        value = np.zeros(np.shape(x), dtype=complex)
        value[inside] = np.exp(-(x[inside]**2 + y[inside]**2)/w**2)*np.exp(1j*phases[row[inside], col[inside]])
        return value
    return field


def pixel_centres(n, pixel_m):
    """Output targets at the physical pixel centres, ordered row-major with +y up."""
    c = (np.arange(n) - (n - 1)/2)*pixel_m
    xx, yy = np.meshgrid(c, -c, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def asm_at_centres(phases, pixel_m, wavelength_m, distance_m, envelope_width_m, refinement, padding,
                   filter_mode="none"):
    """Propagate the point-sampled screen with the core kernel and read pixel centres."""
    n = phases.shape[0]
    optics = Optics(n=n, pixel_m=pixel_m, wavelength_m=wavelength_m, distances_m=(distance_m,),
                    padding=padding, refinement=refinement, boundary="transparent",
                    filter_mode=filter_mode)
    dummy = np.zeros((2, n, n)); dummy[0, 0, 0] = 1; dummy[1, 0, 1] = 1
    model = DiffractiveNetwork(optics, dummy)
    r, pitch = refinement, pixel_m/refinement
    k = (np.arange(n*r) - (n*r - 1)/2)*pitch
    xx, yy = np.meshgrid(k, -k, indexing="xy")
    field = np.zeros((optics.m, optics.m), dtype=complex)
    field[model.roi] = screen_function(phases, pixel_m, envelope_width_m)(xx, yy)
    out = model.propagate(field, model.kernels[0])[model.roi]
    centre = (r - 1)//2
    return out[centre::r, centre::r].ravel()


def pixel_reference(n=16, pixel_m=4e-4, wavelength_m=7.5e-4, distance_m=0.018, envelope_fraction=0.25,
                    refinements=(1, 3, 9), paddings=(8, 16), quadratures=(1, 3, 9, 27), seed=0,
                    phases=None, block=16, filter_modes=("none", "phase_gradient")):
    """Compare ASM(r, p) and RS(q) at pixel centres for one fixed random or supplied screen."""
    refinements, quadratures = _check_odd(refinements, "refinements"), _check_odd(quadratures, "quadratures")
    if len(quadratures) < 2:
        raise ValueError("at least two quadrature levels are required for a convergence check")
    if not isinstance(n, int) or n < 4 or n % 2:
        raise ValueError("n must be an even integer >= 4")
    if phases is None:
        phases = np.random.default_rng(seed).uniform(0, 2*np.pi, (n, n))
    phases = np.asarray(phases, float)
    if phases.shape != (n, n):
        raise ValueError("supplied phases must have shape (n, n)")
    width = envelope_fraction*n*pixel_m
    source = screen_function(phases, pixel_m, width)
    targets = pixel_centres(n, pixel_m)
    rs = {q: rayleigh_sommerfeld(source, n*pixel_m, pixel_m/q, targets, wavelength_m, distance_m, block=block)
          for q in quadratures}
    best = rs[quadratures[-1]]
    values = dict(
        rs_quadrature={f"q{a}_vs_q{b}": relative(rs[a], rs[b]) for a, b in zip(quadratures[:-1], quadratures[1:])},
        rs_quadrature_vs_finest={f"q{q}": relative(rs[q], best) for q in quadratures[:-1]},
        asm={})
    for mode in filter_modes:
        for r in refinements:
            for p in paddings:
                asm = asm_at_centres(phases, pixel_m, wavelength_m, distance_m, width, r, p, mode)
                entry = dict(field_vs_rs_finest=relative(asm, best),
                             intensity_vs_rs_finest=relative(np.abs(asm)**2, np.abs(best)**2))
                if r in rs:
                    entry["field_vs_rs_same_quadrature"] = relative(asm, rs[r])
                values["asm"][f"{mode}_r{r}_p{p}"] = entry
    values["scope"] = ("single pixelated phase screen, scalar RS1 reference, point values at pixel centres; "
                       "not a multilayer or vector validation")
    values["configuration"] = dict(n=n, pixel_m=pixel_m, wavelength_m=wavelength_m, distance_m=distance_m,
                                   envelope_width_m=width, seed=seed, refinements=refinements,
                                   paddings=list(paddings), quadratures=quadratures,
                                   filter_modes=list(filter_modes),
                                   pixel_over_wavelength=pixel_m/wavelength_m)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refinements", type=int, nargs="+", default=[1, 3, 9])
    parser.add_argument("--paddings", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--quadratures", type=int, nargs="+", default=[1, 3, 9, 27])
    parser.add_argument("--filter-modes", nargs="+", default=["none", "phase_gradient"],
                        choices=["none", "phase_gradient"])
    parser.add_argument("--phases", type=Path, help="NPZ with a 'phases' array; layer 0 centre crop is used")
    args = parser.parse_args()
    phases = None
    if args.phases:
        with np.load(args.phases, allow_pickle=False) as data:
            full = np.asarray(data["phases"])
        full = full[0] if full.ndim == 3 else full
        off = (full.shape[0] - args.n)//2
        if off < 0:
            parser.error("--n exceeds the stored phase support")
        phases = full[off:off+args.n, off:off+args.n]
    result = pixel_reference(n=args.n, seed=args.seed, refinements=args.refinements,
                             paddings=args.paddings, quadratures=args.quadratures, phases=phases,
                             filter_modes=tuple(args.filter_modes))
    record(args.output, "pixelated-screen RS reference", result["configuration"], result)
    print(args.output)


if __name__ == "__main__":
    main()
