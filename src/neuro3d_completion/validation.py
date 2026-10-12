"""Executable scalar references and fixed-mask boundary convergence diagnostics."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np

from .core import Optics, DiffractiveNetwork, detector_masks
from .evidence import record, digest_file


def relative(a, b):
    """Compare arrays without removing phase, translating, or renormalizing them."""
    denominator = np.linalg.norm(b)
    if denominator == 0:
        return float(np.linalg.norm(a-b))
    return float(np.linalg.norm(a-b)/denominator)


def rayleigh_sommerfeld(source_fn, source_window_m, source_pitch_m,
                       targets_xy, wavelength_m, distance_m, block=64):
    """Evaluate the exact scalar RS1 kernel by midpoint quadrature.

    source_fn returns physical field amplitudes. Results are point values in
    the same convention; no phase fitting or output normalization is applied.
    Quadrature and source-window convergence are separate validation gates.
    """
    if min(source_window_m, source_pitch_m, wavelength_m, distance_m) <= 0:
        raise ValueError("RS physical parameters must be positive")
    count = int(round(source_window_m/source_pitch_m))
    if count < 2 or not np.isclose(count*source_pitch_m, source_window_m):
        raise ValueError("source window must contain an integer number of pixels")
    coordinates = (np.arange(count)-(count-1)/2)*source_pitch_m
    xx, yy = np.meshgrid(coordinates, -coordinates, indexing="xy")
    source = np.asarray(source_fn(xx, yy), dtype=complex)
    if source.shape != xx.shape or not np.isfinite(source).all():
        raise ValueError("source_fn must return a finite field on the source grid")
    targets = np.asarray(targets_xy, float)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
        raise ValueError("targets_xy must have shape (targets,2)")
    k, result = 2*np.pi/wavelength_m, np.empty(len(targets), dtype=complex)
    for start in range(0, len(targets), block):
        xy = targets[start:start+block]
        radius = np.sqrt((xy[:, 0, None]-xx.ravel())**2 +
                         (xy[:, 1, None]-yy.ravel())**2 + distance_m**2)
        kernel = distance_m*np.exp(1j*k*radius)*(1-1j*k*radius)/(2*np.pi*radius**3)
        result[start:start+len(xy)] = kernel @ source.ravel()*source_pitch_m**2
    return result


def source_field(n, pitch, wavelength, kind):
    """Create deterministic smooth physical inputs, not random pixel noise."""
    coords = (np.arange(n)-(n-1)/2)*pitch
    xx, yy = np.meshgrid(coords, -coords, indexing="xy")
    width = n*pitch/6
    envelope = np.exp(-((xx*xx+yy*yy)/(width*width))**4)
    if kind == "gaussian":
        u = np.exp(-(xx*xx+yy*yy)/(width*width))
    elif kind == "tilted":
        u = envelope*np.exp(2j*np.pi*0.45*xx/wavelength)
    else:
        raise ValueError("unknown source kind")
    return u.astype(complex)


def boundary_convergence(base, phases=None, inputs=None, batch_size=4, paddings=(2, 4, 8)):
    """Compare unchanged physical phase masks on padding factors 2, 4, and 8."""
    masks = detector_masks(base.n, base.pixel_m)
    if inputs is None:
        fields = [source_field(base.n, base.pixel_m, base.wavelength_m, kind)
                  for kind in ("gaussian", "tilted")]
        fields = np.asarray([u/np.linalg.norm(u) for u in fields])
    else:
        fields = np.asarray(inputs, complex)
        if fields.ndim != 3 or fields.shape[1:] != (base.n, base.n) or len(fields) == 0:
            raise ValueError("convergence inputs must have shape (samples,n,n)")
    seed_phases = DiffractiveNetwork(replace(base, padding=2), masks, seed=0).phases
    if phases is not None:
        seed_phases = np.asarray(phases, float)
    outputs = {}
    if len(paddings) < 2 or sorted(set(paddings)) != list(paddings) or min(paddings) < 2:
        raise ValueError("at least two increasing distinct padding factors >= 2 are required")
    for padding in paddings:
        model = DiffractiveNetwork(replace(base, padding=padding), masks, seed=0)
        model.phases = seed_phases.copy()
        field_parts, score_parts = [], []
        for start in range(0, len(fields), batch_size):
            out = model.forward(fields[start:start+batch_size])
            field_parts.append(out["f"][(slice(None), *model.roi)].copy())
            score_parts.append(out["scores"])
        central, scores = np.concatenate(field_parts), np.concatenate(score_parts)
        outputs[padding] = dict(field=central, intensity=np.abs(central)**2,
                                scores=scores, labels=np.argmax(scores, axis=1))
    values = {}
    for small, large in zip(paddings[:-1], paddings[1:]):
        a, b = outputs[small], outputs[large]
        values[f"padding_{small}_vs_{large}"] = dict(
            field_relative_L2=relative(a["field"], b["field"]),
            intensity_relative_L2=relative(a["intensity"], b["intensity"]),
            score_relative_L2=relative(a["scores"], b["scores"]),
            changed_label_fraction=float(np.mean(a["labels"] != b["labels"])))
    return values


def pixel_convergence(base, phases=None, inputs=None, refinements=(1, 2, 4)):
    """Refine quadrature while preserving piecewise-constant physical phase cells.

    Numerical samples of each physical phase are repeated, never retrained.
    Intensities are coarsened by block sums, fields by cell-average projection.
    Scores use detector rectangles of unchanged physical dimensions.
    """
    if sorted(set(refinements)) != list(refinements) or len(refinements) < 2 or refinements[0] != 1:
        raise ValueError("increasing refinements beginning at 1 are required")
    masks = detector_masks(base.n, base.pixel_m)
    initial = DiffractiveNetwork(base, masks, seed=0)
    fixed_phases = initial.phases if phases is None else np.asarray(phases, float)
    if inputs is None:
        inputs = np.asarray([source_field(base.n, base.pixel_m, base.wavelength_m, kind)
                             for kind in ("gaussian", "tilted")])
        inputs /= np.linalg.norm(inputs, axis=(1, 2))[:, None, None]
    inputs = np.asarray(inputs, complex)
    results = {}
    for r in refinements:
        config = replace(base, refinement=base.refinement*r)
        model = DiffractiveNetwork(config, masks, seed=0)
        model.phases = fixed_phases.copy()
        sampled = inputs
        field_parts, score_parts = [], []
        for i in range(len(sampled)):
            out = model.forward(sampled[i])
            field_parts.append(out["f"][model.roi]); score_parts.append(out["scores"])
        fields = np.asarray(field_parts)
        total_refinement = config.refinement
        shape = (len(sampled), base.n, total_refinement, base.n, total_refinement)
        results[r] = dict(field=fields.reshape(shape).sum(axis=(2, 4))/total_refinement,
                          intensity=(np.abs(fields)**2).reshape(shape).sum(axis=(2, 4)),
                          scores=np.asarray(score_parts))
    values = {}
    for a, b in zip(refinements[:-1], refinements[1:]):
        values[f"refinement_{a}_vs_{b}"] = {key+"_relative_L2": relative(results[a][key], results[b][key])
                                             for key in ("field", "intensity", "scores")}
    values["independent_physical_phase_parameters"] = int(fixed_phases.size)
    values["interpretation"] = "piecewise-constant physical pixels; cell-average projection; no retraining"
    return values


def scalar_reference(n=32, pitch=0.0004, wavelength=0.00075, distance=0.018):
    """Check ASM against independently sampled RS, preserving physical coordinates."""
    width = n*pitch/8
    def gaussian(x, y):
        return np.exp(-(x*x+y*y)/width**2).astype(complex)
    coordinates = (np.arange(n)-(n-1)/2)*pitch
    xx, yy = np.meshgrid(coordinates, -coordinates, indexing="xy")
    targets = np.column_stack((xx.ravel(), yy.ravel()))
    window = 2*n*pitch
    rs_coarse = rayleigh_sommerfeld(gaussian, window, pitch/2, targets, wavelength, distance)
    rs_fine = rayleigh_sommerfeld(gaussian, window, pitch/4, targets, wavelength, distance)
    rs_wide = rayleigh_sommerfeld(gaussian, 2*window, pitch/4, targets, wavelength, distance)
    optics = Optics(n=n, pixel_m=pitch, wavelength_m=wavelength,
                    distances_m=(distance, distance), padding=4, boundary="transparent")
    dummy_masks = np.zeros((2, n, n)); dummy_masks[0, n//2, n//2] = 1
    dummy_masks[1, n//2, n//2+1] = 1
    model = DiffractiveNetwork(optics, dummy_masks)
    padded = np.zeros((optics.m, optics.m), complex)
    padded[model.roi] = gaussian(xx, yy)
    propagated = model.propagate(padded, model.kernels[0])[model.roi].ravel()
    return dict(quadrature_relative_L2=relative(rs_coarse, rs_fine),
                source_window_relative_L2=relative(rs_fine, rs_wide),
                ASM_vs_RS_complex_relative_L2=relative(propagated, rs_wide),
                source="smooth Gaussian; no learned layers", n=n,
                reference_scope="independent scalar reference, not vector Maxwell validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["boundary", "pixel", "reference"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--boundary", choices=["aperture", "transparent"], default="aperture")
    parser.add_argument("--phases", type=Path)
    parser.add_argument("--dev-split", type=Path)
    parser.add_argument("--paddings", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--refinements", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--base-padding", type=int, default=2)
    parser.add_argument("--quadrature", type=int, default=1)
    args = parser.parse_args()
    if args.command in {"boundary", "pixel"}:
        phases = None
        if args.phases:
            with np.load(args.phases, allow_pickle=False) as data:
                phases = data["phases"]
        config = Optics(n=args.n, boundary=args.boundary,
                        padding=args.base_padding, refinement=args.quadrature)
        inputs = None
        if args.dev_split:
            from .data import encode
            with np.load(args.dev_split, allow_pickle=False) as data:
                images = encode(data["images"])
            off = (args.n-64)//2
            if off < 0 or 2*off+64 != args.n:
                raise ValueError("DEV image support does not align with the physical grid")
            inputs = np.pad(images, ((0, 0), (off, off), (off, off)))
        if args.command == "boundary":
            result = boundary_convergence(config, phases, inputs=inputs, paddings=args.paddings)
        else:
            result = pixel_convergence(config, phases, inputs=inputs, refinements=args.refinements)
        result["input_scope"] = "supplied DEV split" if inputs is not None else "two smooth diagnostic inputs"
        if args.dev_split:
            result["dev_split_sha256"] = digest_file(args.dev_split)
        if args.phases:
            result["phases_sha256"] = digest_file(args.phases)
        record(args.output, "exploratory "+args.command+" convergence", config, result)
    else:
        result = scalar_reference(n=args.n)
        record(args.output, "scalar RS reference", dict(n=args.n), result)
    print(args.output)


if __name__ == "__main__":
    main()
