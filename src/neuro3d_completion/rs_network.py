"""Direct-summation Rayleigh-Sommerfeld reference for a complete multilayer phase network.

No FFT, no periodic window and no spectral filter is used. Every propagation is the
scalar RS1 integral evaluated by midpoint quadrature on q x q subcells per physical
pixel, aligned with pixel boundaries. The masks are piecewise constant, the exterior is
absorbing (the core 'aperture' convention), and the input is the core cell-energy
amplitude u, i.e. a piecewise-constant physical field E = u / pixel.

Cost per propagation is (n q)^4 complex kernel evaluations. n=16, q=9 runs on a CPU in
seconds to minutes; n=128, q=3 (about 2.2e10 evaluations per propagation) needs a GPU.
The quadrature converges as O(q^-2) for piecewise-smooth integrands; convergence must
be demonstrated with at least three q values, never assumed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import torch

from .core import Optics, DiffractiveNetwork, positive
from .evidence import record
from .guards import frozen_optics, refuse_reserved_split
from .validation import relative


def subcell_centres(n, pixel_m, q):
    """Coordinates of q x q subcell centres over an n x n support, +y up, row-major."""
    c = (np.arange(n*q) - (n*q - 1)/2)*pixel_m/q
    xx, yy = np.meshgrid(c, -c, indexing="xy")
    return xx.ravel(), yy.ravel()


def _rs_apply(source, xs, ys, xt, yt, distance, wavelength, area, block, device):
    """E(t) = sum_s K(t - s) E(s) area, with the exact RS1 kernel, in target blocks."""
    k = 2*np.pi/wavelength
    xs_t = torch.as_tensor(xs, dtype=torch.float64, device=device)
    ys_t = torch.as_tensor(ys, dtype=torch.float64, device=device)
    src = torch.as_tensor(source, dtype=torch.complex128, device=device)
    out = torch.empty(len(xt), dtype=torch.complex128, device=device)
    for start in range(0, len(xt), block):
        tx = torch.as_tensor(xt[start:start+block], dtype=torch.float64, device=device)[:, None]
        ty = torch.as_tensor(yt[start:start+block], dtype=torch.float64, device=device)[:, None]
        radius = torch.sqrt((tx - xs_t)**2 + (ty - ys_t)**2 + distance**2)
        phase = k*radius
        amplitude = distance/(2*np.pi*radius**3)
        # e^{ikR}(1 - ikR) = (cos + i sin)(1 - i kR)
        real = amplitude*(torch.cos(phase) + phase*torch.sin(phase))
        imag = amplitude*(torch.sin(phase) - phase*torch.cos(phase))
        kernel = torch.complex(real, imag)
        out[start:start+block] = kernel @ src
    return (out*area).cpu().numpy()


def rs_network_fields(inputs, phases, pixel_m, wavelength_m, distances_m, q, block=2048, device="cpu"):
    """Return the detector-plane physical field at q x q subcells for each input.

    inputs: (batch, n, n) cell-energy amplitudes; phases: (layers, n, n);
    distances_m: layers + 1 positive distances. Absorbing exterior at every plane.
    """
    u = np.asarray(inputs, dtype=complex)
    phases = np.asarray(phases, dtype=float)
    if u.ndim == 2:
        u = u[None]
    a, lam = positive(pixel_m, "pixel_m"), positive(wavelength_m, "wavelength_m")
    if (u.ndim != 3 or phases.ndim != 3 or u.shape[1:] != phases.shape[1:] or
            len(distances_m) != len(phases) + 1 or not np.isfinite(u).all() or not np.isfinite(phases).all()):
        raise ValueError("inconsistent inputs, phases or distances")
    if not isinstance(q, int) or q < 1:
        raise ValueError("q must be a positive integer")
    n = u.shape[1]
    xs, ys = subcell_centres(n, a, q)
    area = (a/q)**2
    expand = lambda x: np.repeat(np.repeat(x, q, axis=-2), q, axis=-1).reshape(*x.shape[:-2], -1)
    masks = np.exp(1j*expand(phases))
    fields = []
    for sample in u:
        e = expand(sample/a)
        for layer, d in enumerate(distances_m):
            e = _rs_apply(e, xs, ys, xs, ys, positive(d, "distance"), lam, area, block, device)
            if layer < len(phases):
                e = e*masks[layer]
        fields.append(e.reshape(n*q, n*q))
    return np.asarray(fields)


def rs_scores(fields, masks, pixel_m, q, beta):
    """Integrate |E|^2 over each detector region with the same subcells (core score units)."""
    m = np.repeat(np.repeat(np.asarray(masks, float), q, axis=1), q, axis=2)
    return beta*np.einsum("kij,bij->bk", m, np.abs(fields)**2)*(pixel_m/q)**2


def compare_with_core(n=16, layers=3, pixel_m=4e-4, wavelength_m=7.5e-4, distance_m=0.018, samples=4,
                      quadratures=(1, 3, 9), core_settings=((3, 8), (3, 16), (3, 32)), seed=0,
                      filter_mode="none", device="cpu", block=2048):
    """Small-scale study: RS(q) convergence and core ASM(r, p) error on detector scores."""
    if n < 8 or n % 2:
        raise ValueError("n must be an even integer >= 8")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2*np.pi, (layers, n, n))
    # Smooth positive inputs inside the central half of the support, unit cell energy.
    c = np.arange(n) - (n - 1)/2
    yy, xx = np.meshgrid(c, c, indexing="ij")
    inputs = []
    for _ in range(samples):
        x0, y0 = rng.uniform(-n/8, n/8, 2)
        blob = np.exp(-((xx - x0)**2 + (yy - y0)**2)/(n/6)**2)*(1 + 0.5*rng.uniform(size=(n, n)))
        blob[(np.abs(xx) > n/4) | (np.abs(yy) > n/4)] = 0
        inputs.append(blob/np.linalg.norm(blob))
    inputs = np.asarray(inputs)
    # Four disjoint 2x2-pixel detectors around the centre (classes 0..3).
    masks = np.zeros((4, n, n))
    h = n//2
    for k, (r0, c0) in enumerate(((h-3, h-3), (h-3, h+1), (h+1, h-3), (h+1, h+1))):
        masks[k, r0:r0+2, c0:c0+2] = 1
    distances = (distance_m,)*(layers + 1)
    t0 = time.perf_counter()
    rs = {}
    for q in quadratures:
        rs[q] = rs_scores(rs_network_fields(inputs, phases, pixel_m, wavelength_m, distances, q, block, device),
                          masks, pixel_m, q, beta=1.0)
    rs_time = time.perf_counter() - t0
    best = rs[quadratures[-1]]
    labels = best.argmax(axis=1)
    values = dict(rs_quadrature={f"q{a}_vs_q{b}": relative(rs[a], rs[b])
                                 for a, b in zip(quadratures[:-1], quadratures[1:])},
                  core={}, rs_seconds=rs_time)
    for r, p in core_settings:
        optics = Optics(n=n, pixel_m=pixel_m, wavelength_m=wavelength_m, distances_m=distances,
                        padding=p, refinement=r, boundary="aperture", filter_mode=filter_mode)
        model = DiffractiveNetwork(optics, masks, beta=1.0)
        model.phases = phases.copy()
        scores = model.forward(inputs)["scores"]
        values["core"][f"r{r}_p{p}"] = dict(scores_vs_rs_finest=relative(scores, best),
                                            label_agreement=float(np.mean(scores.argmax(axis=1) == labels)))
    values["configuration"] = dict(n=n, layers=layers, pixel_m=pixel_m, wavelength_m=wavelength_m,
                                   distance_m=distance_m, samples=samples, quadratures=list(quadratures),
                                   core_settings=[list(x) for x in core_settings], seed=seed,
                                   filter_mode=filter_mode, device=device)
    values["scope"] = ("random phase network, smooth synthetic inputs, scalar RS1 direct summation; "
                       "small-scale verification of the discretisation, not a statement about trained masks")
    return values


def frozen_comparison(phases, optics, masks, inputs, quadratures=(1, 3), block=256, device="cpu"):
    """RS scores of frozen phases on given inputs versus the core at the frozen optics.

    'optics' is the frozen numerical configuration; RS uses only its physical part
    (pixel, wavelength, distances) and the absorbing exterior. Use DEV inputs only.
    """
    if optics.boundary != "aperture":
        raise ValueError("the direct RS reference implements the absorbing exterior only")
    model = DiffractiveNetwork(optics, masks, beta=1.0)
    model.phases = np.asarray(phases, float)
    core = model.forward(inputs)["scores"]
    rs = {q: rs_scores(rs_network_fields(inputs, phases, optics.pixel_m, optics.wavelength_m, optics.distances_m,
                                         q, block, device), masks, optics.pixel_m, q, beta=1.0)
          for q in quadratures}
    best = rs[quadratures[-1]]
    return dict(core_vs_rs_finest=relative(core, best),
                core_label_agreement=float(np.mean(core.argmax(axis=1) == best.argmax(axis=1))),
                rs_quadrature={f"q{a}_vs_q{b}": relative(rs[a], rs[b]) for a, b in zip(quadratures[:-1], quadratures[1:])},
                rs_scores={f"q{q}": rs[q].tolist() for q in quadratures}, core_scores=core.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-phases", type=Path, help="phases.npz from benchmark.final (with architecture.json)")
    parser.add_argument("--split", type=Path, help="DEV split NPZ for --frozen-phases; TEST is refused")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--quadratures", type=int, nargs="+", default=[1, 3, 9])
    parser.add_argument("--core", type=int, nargs="+", default=[3, 8, 3, 16, 3, 32],
                        help="flat list of refinement/padding pairs")
    parser.add_argument("--filter-mode", choices=["none", "phase_gradient"], default="none")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.frozen_phases:
        import json
        from .core import detector_masks
        from .data import encode
        if args.split is None:
            parser.error("--frozen-phases needs a DEV --split")
        refuse_reserved_split(args.split)
        optics, source = frozen_optics(args.frozen_phases, padding=4)
        stored = dict(optics.__dict__, optics_source=source)
        with np.load(args.frozen_phases, allow_pickle=False) as data:
            phases = np.asarray(data["phases"], float)
        with np.load(args.split, allow_pickle=False) as data:
            images = data["images"][args.indices]
        x = encode(images)
        off = (optics.n - x.shape[-1])//2
        x = np.pad(x, ((0, 0), (off, off), (off, off)))
        result = frozen_comparison(phases, optics, detector_masks(optics.n, optics.pixel_m), x,
                                   tuple(args.quadratures), args.block, args.device)
        result["indices"] = args.indices
        record(args.output, "frozen configuration versus direct RS", stored, result)
        print(args.output)
        return
    if len(args.core) % 2:
        parser.error("--core needs refinement/padding pairs")
    pairs = tuple(zip(args.core[0::2], args.core[1::2]))
    result = compare_with_core(n=args.n, samples=args.samples, quadratures=tuple(args.quadratures),
                               core_settings=pairs, filter_mode=args.filter_mode, device=args.device,
                               block=args.block)
    record(args.output, "multilayer RS direct-summation reference", result["configuration"], result)
    print(args.output)


if __name__ == "__main__":
    main()
