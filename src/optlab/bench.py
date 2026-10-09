# SPDX-License-Identifier: Apache-2.0
"""Declarative optical bench (JSON "optlab.bench/1") and simulation.

The bench is a source plane at z_m, followed by layers (thin phase elements) and a
detector plane. Distances are differences of consecutive z_m values, and the z_m
values must be strictly increasing. Free-space propagation uses the angular-spectrum
method with time dependence exp(-i*w*t). Each layer multiplies the field by

    t = exp(i*2*pi*(n_mat - n_med)*h/lambda) * exp(-2*pi*kappa*h/lambda),
    h = height_npy * height_max_m,

where lambda = wavelength_m is the vacuum wavelength, n_mat + i*kappa is the complex
index of the layer material, and n_med is the layer's n_medium. Relative paths are
resolved against base_dir.

medium_index (default 1.0) is the refractive index of the medium between the planes.
Propagation uses wavelength_m / medium_index, and the n_medium of every layer must
equal medium_index.
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .elements import detector_masks, phase_element
from .propagation import angular_spectrum

SCHEMA = "optlab.bench/1"


def _need(d, key, where):
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"missing '{key}' in {where}")
    return d[key]


def _num(v, name, positive=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
        raise ValueError(f"'{name}' must be a finite number")
    if positive and v <= 0:
        raise ValueError(f"'{name}' must be > 0")
    return float(v)


def load_bench(path):
    """Load a bench JSON file, validate it, and return the bench dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return _validate(json.load(fh))


def _validate(b):
    if not isinstance(b, dict):
        raise ValueError("bench must be a JSON object")
    if b.get("schema") != SCHEMA:
        raise ValueError(f"schema must be '{SCHEMA}'")
    _num(_need(b, "wavelength_m", "bench"), "wavelength_m", True)
    medium = _num(b.get("medium_index", 1.0), "medium_index", True)
    grid = _need(b, "grid", "bench")
    n = _need(grid, "n", "grid")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("grid.n must be an integer >= 2")
    _num(_need(grid, "pixel_m", "grid"), "grid.pixel_m", True)
    src = _need(b, "source", "bench")
    zs = _num(_need(src, "z_m", "source"), "source.z_m")
    layers = b.get("layers", [])
    if not isinstance(layers, list):
        raise ValueError("layers must be a list")
    det = _need(b, "detector", "bench")
    zd = _num(_need(det, "z_m", "detector"), "detector.z_m")
    regions = det.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("detector.regions must be a list")
    for i, r in enumerate(regions):
        for k in ("cx_m", "cy_m", "w_m", "h_m"):
            v = _num(_need(r, k, f"detector.regions[{i}]"), k)
            if k in ("w_m", "h_m") and v <= 0:
                raise ValueError(f"detector.regions[{i}].{k} must be > 0")
    prev, prev_name = zs, "source"
    for i, ly in enumerate(layers):
        name = ly.get("name", f"layers[{i}]") if isinstance(ly, dict) else f"layers[{i}]"
        z = _num(_need(ly, "z_m", name), f"{name}.z_m")
        if z <= prev:
            raise ValueError(f"z must be strictly increasing: {name} z_m={z} <= {prev_name} z_m={prev}")
        prev, prev_name = z, name
        nm = _num(_need(ly, "n_medium", name), "n_medium", True)
        if not np.isclose(nm, medium, rtol=0.0, atol=1e-12):
            raise ValueError(f"{name}: n_medium={nm} != bench medium_index={medium}")
        _num(_need(ly, "n_material", name), "n_material", True)
        if _num(ly.get("kappa", 0.0), "kappa") < 0:
            raise ValueError(f"{name}: kappa must be >= 0")
        if not isinstance(_need(ly, "height_npy", name), str):
            raise ValueError(f"{name}: height_npy must be a path string")
        if _num(_need(ly, "height_max_m", name), "height_max_m") < 0:
            raise ValueError(f"{name}: height_max_m must be >= 0")
    if zd <= prev:
        raise ValueError(f"z must be strictly increasing: detector z_m={zd} <= {prev_name} z_m={prev}")
    return b


def _propagation_segments(b):
    """Return (label, length_m) for each free-space segment: source -> layer 1, ..., layer N -> detector."""
    segments = []
    z, prev = b["source"]["z_m"], "source"
    for i, ly in enumerate(b.get("layers", [])):
        name = ly.get("name", f"layers[{i}]")
        segments.append((f"{prev} -> {name}", ly["z_m"] - z))
        z, prev = ly["z_m"], name
    segments.append((f"{prev} -> detector", b["detector"]["z_m"] - z))
    return segments


def simulate_bench(path_or_dict, base_dir=None, output_path=None, return_scores=False,
                   allow_out_of_domain=False):
    """Return the intensity array I (n, n) at the detector.

    With return_scores=True, return (I, {label: energy_in_region}). The energy of a
    region is the unnormalized sum of I over the pixels whose centers lie inside it.
    Keys are the region "label", or its index when no label is given. Region
    coordinates (cx_m, cy_m, w_m, h_m) are in meters from the grid center, with +x to
    the right and +y up. A region is the half-open rectangle
    [cx_m - w_m/2, cx_m + w_m/2) x [cy_m - h_m/2, cy_m + h_m/2).

    Validity domain: the angular-spectrum propagation is applied to free-space segments
    only while their length d is below z_max = N * pixel_m**2 * n_medium / wavelength_m,
    where N is the grid size and the wavelength in the medium is wavelength_m /
    medium_index. This sampling-based criterion is the one checked below; it is not a
    comparison with an independent solution. Unless allow_out_of_domain=True, a segment
    with d >= z_max raises ValueError naming the segment, its length and z_max. With
    allow_out_of_domain=True, each such segment instead emits a warnings.warn with the
    same information and the simulation continues; the result then lies outside the
    stated validity domain and must be reported as such.
    """
    if isinstance(path_or_dict, dict):
        b = _validate(path_or_dict)
    else:
        b = load_bench(path_or_dict)
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(path_or_dict))
    base_dir = base_dir or os.getcwd()

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    lam = b["wavelength_m"]
    n = b["grid"]["n"]
    px = b["grid"]["pixel_m"]
    src = b["source"]
    if src.get("amplitude_npy"):
        u = np.load(rp(src["amplitude_npy"])).astype(complex)
        if u.shape != (n, n):
            raise ValueError(f"amplitude_npy shape {u.shape} != ({n},{n})")
    else:
        u = np.ones((n, n), dtype=complex)
    z = src["z_m"]
    lam_med = lam / float(b.get("medium_index", 1.0))  # wavelength in the medium
    z_max = n * px ** 2 / lam_med  # validity domain of the angular-spectrum propagation (see the docstring)
    problems = [
        f"propagation segment '{label}' has length {d:.6g} m >= z_max = {z_max:.6g} m "
        f"(z_max = N*pixel_m**2*n_medium/wavelength_m, N={n}): out of domain"
        for label, d in _propagation_segments(b) if d >= z_max
    ]
    if problems and not allow_out_of_domain:
        raise ValueError("; ".join(problems))
    for msg in problems:
        warnings.warn(msg, stacklevel=2)
    for ly in b.get("layers", []):
        u = angular_spectrum(u, px, lam_med, ly["z_m"] - z)
        z = ly["z_m"]
        h = np.load(rp(ly["height_npy"]))
        if h.shape != (n, n):
            raise ValueError(f"{ly.get('name', '?')}: height shape {h.shape} != ({n},{n})")
        if not np.isfinite(h).all():
            raise ValueError(f"{ly.get('name', '?')}: height contains NaN/inf")
        h = h.astype(float) * ly["height_max_m"]
        u = u * phase_element(h, lam, ly["n_material"], ly["n_medium"], ly.get("kappa", 0.0))
    u = angular_spectrum(u, px, lam_med, b["detector"]["z_m"] - z)
    I = np.abs(u) ** 2
    if output_path:
        np.save(output_path, I)
    if return_scores:
        regs = b["detector"].get("regions", [])
        masks = detector_masks(n, px, regs)
        scores = {r.get("label", i): float((masks[i] * I).sum()) for i, r in enumerate(regs)}
        return I, scores
    return I
