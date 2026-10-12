"""Fabrication and alignment perturbations of frozen phase masks (TRAIN/DEV analysis only).

Every perturbation is applied to a copy; the frozen model is never modified. Physical
interpretations: phase quantisation (discrete height levels), Gaussian phase error
(height error sigma_h gives sigma_phi = 2*pi*dn*sigma_h/lambda), lateral layer shift in
numerical samples, axial distance error, and wavelength detuning without dispersion
(phi' = (phi mod 2pi) * lambda / lambda', i.e. fixed fabricated heights). Distributions
must come from measured or specified tolerances, fixed before the analysis.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np

from .core import DiffractiveNetwork, Optics, detector_masks
from .evidence import digest_file, record
from .guards import frozen_optics, refuse_reserved_split


def forward_custom(model, inputs, transmissions, kernels):
    """Batched forward pass with supplied transmissions and kernels (core conventions)."""
    o = model.optics
    u = np.asarray(inputs, dtype=complex)
    if u.ndim == 2:
        u = u[None]
    if u.ndim != 3 or u.shape[1:] != (o.n, o.n) or not np.isfinite(u).all():
        raise ValueError("inputs must be finite (batch, n, n)")
    t = np.asarray(transmissions)
    if t.shape != (o.layers, o.m, o.m) or len(kernels) != o.layers + 1:
        raise ValueError("transmissions or kernels do not match the optics")
    r = o.refinement
    a = np.zeros((len(u), o.m, o.m), dtype=complex)
    a[(slice(None), *model.roi)] = np.repeat(np.repeat(u, r, axis=1), r, axis=2)/r
    for k in range(o.layers):
        a = model.propagate(a, kernels[k])*t[k]
    f = model.propagate(a, kernels[-1])
    if o.boundary == "aperture":
        f = f*model.aperture
    return model.beta*np.einsum("kij,bij->bk", model.full_masks, np.abs(f)**2)


def quantize(phases, levels):
    """Round phases modulo 2*pi to 'levels' equally spaced values."""
    if not isinstance(levels, int) or levels < 2:
        raise ValueError("levels must be an integer >= 2")
    step = 2*np.pi/levels
    return (np.round(np.mod(phases, 2*np.pi)/step) % levels)*step


def shifted_transmissions(model, shifts):
    """Roll each layer's transmission by (dy, dx) numerical samples inside the padded window."""
    t = model.transmissions()
    o = model.optics
    margin = (o.m - o.n*o.refinement)//2
    if len(shifts) != o.layers:
        raise ValueError("one (dy, dx) shift per layer is required")
    out = np.empty_like(t)
    for k, (dy, dx) in enumerate(shifts):
        if max(abs(int(dy)), abs(int(dx))) > margin:
            raise ValueError("shift exceeds the padding margin; increase padding")
        out[k] = np.roll(t[k], (int(dy), int(dx)), axis=(0, 1))
    return out


def perturbed_scores(model, inputs, kind, value, rng=None):
    """Scores of a perturbed copy. kind in {none, quantize, phase_noise, shift, distance, wavelength}."""
    work = DiffractiveNetwork(model.optics, model.masks, beta=model.beta)
    work.phases = model.phases.copy()
    kernels = list(model.kernels)
    if kind == "none":
        pass
    elif kind == "quantize":
        work.phases = quantize(work.phases, int(value))
    elif kind == "phase_noise":
        if rng is None:
            raise ValueError("phase_noise requires an explicit random generator")
        work.phases = work.phases + rng.normal(0, float(value), work.phases.shape)
    elif kind == "shift":
        return forward_custom(work, inputs, shifted_transmissions(work, value), kernels)
    elif kind == "distance":
        deltas = np.asarray(value, float)
        if deltas.shape != (model.optics.layers + 1,):
            raise ValueError("one distance error per propagation is required")
        kernels = [work._kernel(d + e) for d, e in zip(model.optics.distances_m, deltas)]
    elif kind == "wavelength":
        lam = float(value)
        optics = replace(model.optics, wavelength_m=lam)
        work = DiffractiveNetwork(optics, model.masks, beta=model.beta)
        work.phases = np.mod(model.phases, 2*np.pi)*model.optics.wavelength_m/lam
        kernels = list(work.kernels)
    else:
        raise ValueError("unknown perturbation kind")
    return forward_custom(work, inputs, work.transmissions(), kernels)


def sweep(model, inputs, labels, specification, seed=0, repeats=5):
    """Accuracy, label flips and score change for each (kind, value) in the specification."""
    y = np.asarray(labels)
    base = forward_custom(model, inputs, model.transmissions(), model.kernels)
    base_labels = base.argmax(axis=1)
    rng = np.random.default_rng(seed)
    rows = []
    for kind, value in specification:
        count = repeats if kind == "phase_noise" else 1
        for repeat in range(count):
            scores = perturbed_scores(model, inputs, kind, value, rng)
            rows.append(dict(kind=kind, value=value if np.isscalar(value) else np.asarray(value).tolist(),
                             repeat=repeat, accuracy=float(np.mean(scores.argmax(axis=1) == y)),
                             label_flip_fraction=float(np.mean(scores.argmax(axis=1) != base_labels)),
                             score_relative_L2=float(np.linalg.norm(scores - base)/np.linalg.norm(base))))
    return dict(baseline_accuracy=float(np.mean(base_labels == y)), rows=rows)


DEFAULT_SPECIFICATION = ([("quantize", q) for q in (2, 4, 8, 16, 64)] +
                         [("phase_noise", s) for s in (0.05, 0.1, 0.2, 0.4)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="TRAIN or DEV split NPZ; reserved splits are refused")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--boundary", choices=["aperture", "transparent"], default="aperture")
    parser.add_argument("--filter-mode", choices=["none", "phase_gradient"], default="none")
    parser.add_argument("--beta", type=float, default=100.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refuse_reserved_split(args.split)
    from .data import encode
    with np.load(args.phases, allow_pickle=False) as data:
        phases = data["phases"]
    with np.load(args.split, allow_pickle=False) as data:
        images, labels = data["images"][:args.limit], data["labels"][:args.limit]
    optics, source = frozen_optics(args.phases, args.padding, args.refinement, args.boundary, args.filter_mode)
    n = optics.n
    x = encode(images)
    off = (n - x.shape[-1])//2
    x = np.pad(x, ((0, 0), (off, off), (off, off)))
    model = DiffractiveNetwork(optics, detector_masks(n, optics.pixel_m), beta=args.beta)
    model.phases = np.asarray(phases, float)
    spec = DEFAULT_SPECIFICATION + [("shift", [(s, 0)]*optics.layers) for s in (1, 2)]
    result = sweep(model, x, labels, spec)
    result.update(phases_sha256=digest_file(args.phases), split_sha256=digest_file(args.split), optics_source=source,
                  scope="exploratory TRAIN/DEV robustness of frozen masks; not a confirmatory TEST result")
    record(args.output, "fabrication robustness sweep", optics, result)
    print(args.output)


if __name__ == "__main__":
    main()
