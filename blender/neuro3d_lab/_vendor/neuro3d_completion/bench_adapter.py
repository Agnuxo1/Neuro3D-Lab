"""Adapter for existing optlab.bench/1 JSON files with explicit finite-support optics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .core import Optics, DiffractiveNetwork, detector_masks, positive


def phase_element(height_m, wavelength_m, n_material, n_medium, kappa=0.0):
    """Validate and compute the passive thin-element transmission."""
    h = np.asarray(height_m, dtype=float)
    wavelength = positive(wavelength_m, "wavelength_m")
    material, medium = positive(n_material, "n_material"), positive(n_medium, "n_medium")
    absorption = float(kappa)
    if (not np.isfinite(h).all() or np.any(h < 0) or
            not np.isfinite(absorption) or absorption < 0):
        raise ValueError("finite nonnegative thickness and absorption are required")
    return np.exp(2j*np.pi*(material-medium)*h/wavelength)*np.exp(-2*np.pi*absorption*h/wavelength)


def simulate_bench_v2(path, *, padding=4, refinement=1, boundary="aperture", filter_mode="none", output_path=None):
    """Run an existing bench with finite phase support and no implicit TEST use.

    The legacy entry point remains unchanged. This adapter is a separately
    versioned implementation and must be selected explicitly in Blender.
    """
    from ..optlab.bench import load_bench
    path = Path(path).resolve()
    bench, base = load_bench(path), path.parent
    n, pixel = bench["grid"]["n"], bench["grid"]["pixel_m"]
    wavelength, medium = bench["wavelength_m"], bench.get("medium_index", 1.0)
    layers = bench.get("layers", [])
    zs = [bench["source"]["z_m"]] + [ly["z_m"] for ly in layers] + [bench["detector"]["z_m"]]
    optics = Optics(n=n, pixel_m=pixel, wavelength_m=wavelength/medium,
                    distances_m=tuple(np.diff(zs)), padding=padding, refinement=refinement,
                    boundary=boundary, filter_mode=filter_mode)
    # Dummy detectors support field computation when the bench has no scored regions.
    regions = bench["detector"].get("regions", [])
    if len(regions) >= 2:
        masks = detector_masks(n, pixel, regions)
    else:
        masks = np.zeros((2, n, n)); masks[0, n//2, n//2] = 1; masks[1, n//2, n//2+1] = 1
    engine = DiffractiveNetwork(optics, masks)
    source_path = bench["source"].get("amplitude_npy")
    source = np.ones((n, n), complex) if source_path is None else np.load(base/source_path, allow_pickle=False).astype(complex)
    if source.shape != (n, n) or not np.isfinite(source).all():
        raise ValueError("invalid source field")
    a = np.zeros((optics.m, optics.m), complex)
    a[engine.roi] = np.repeat(np.repeat(source, refinement, axis=0), refinement, axis=1)
    for k, layer in enumerate(layers):
        a = engine.propagate(a, engine.kernels[k])
        height = np.load(base/layer["height_npy"], allow_pickle=False)
        if height.shape != (n, n):
            raise ValueError("height grid does not match physical support")
        transmission = phase_element(height*layer["height_max_m"], wavelength,
                                     layer["n_material"], layer["n_medium"], layer.get("kappa", 0.0))
        full = np.full((optics.m, optics.m), 0 if boundary == "aperture" else 1, complex)
        full[engine.roi] = np.repeat(np.repeat(transmission, refinement, axis=0), refinement, axis=1)
        a *= full
    a = engine.propagate(a, engine.kernels[-1])
    if boundary == "aperture":
        a *= engine.aperture
    fine_field = a[engine.roi]
    coarse_shape = (n, refinement, n, refinement)
    intensity = (np.abs(fine_field)**2).reshape(coarse_shape).mean(axis=(1, 3))
    field = fine_field.reshape(coarse_shape).mean(axis=(1, 3))
    if output_path is not None:
        with Path(output_path).open("xb") as stream:
            np.save(stream, intensity)
    if regions:
        scores_masks = detector_masks(n, pixel, regions)
        scores = {str(r.get("label", i)): float(np.sum(mask*intensity))
                  for i, (r, mask) in enumerate(zip(regions, scores_masks))}
    else:
        scores = {}
    return dict(intensity=intensity, field=field, scores=scores,
                schema="optlab.bench/1", numerical_backend="neuro3d_completion/v1")
