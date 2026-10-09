"""Banco optico declarativo (JSON "optlab.bench/1") y simulacion.

Fuente en z_m -> capas (elementos de fase delgados) -> detector. Las distancias
son diferencias de z (estrictamente crecientes). Propagacion por espectro angular,
exp(-i w t); t = exp(i 2 pi (n_mat-n_med) h/lambda) exp(-2 pi kappa h/lambda),
h = height_npy * height_max_m. Rutas relativas a base_dir.
"""
from __future__ import annotations

import json
import os

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
    with open(path, "r", encoding="utf-8") as fh:
        return _validate(json.load(fh))


def _validate(b):
    if not isinstance(b, dict):
        raise ValueError("bench must be a JSON object")
    if b.get("schema") != SCHEMA:
        raise ValueError(f"schema must be '{SCHEMA}'")
    _num(_need(b, "wavelength_m", "bench"), "wavelength_m", True)
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
            _num(_need(r, k, f"detector.regions[{i}]"), k)
    prev, prev_name = zs, "source"
    for i, ly in enumerate(layers):
        name = ly.get("name", f"layers[{i}]") if isinstance(ly, dict) else f"layers[{i}]"
        z = _num(_need(ly, "z_m", name), f"{name}.z_m")
        if z <= prev:
            raise ValueError(f"z must be strictly increasing: {name} z_m={z} <= {prev_name} z_m={prev}")
        prev, prev_name = z, name
        _num(_need(ly, "n_medium", name), "n_medium", True)
        _num(_need(ly, "n_material", name), "n_material", True)
        if _num(ly.get("kappa", 0.0), "kappa") < 0:
            raise ValueError(f"{name}: kappa must be >= 0")
        if not isinstance(_need(ly, "height_npy", name), str):
            raise ValueError(f"{name}: height_npy must be a path string")
        _num(_need(ly, "height_max_m", name), "height_max_m")
    if zd <= prev:
        raise ValueError(f"z must be strictly increasing: detector z_m={zd} <= {prev_name} z_m={prev}")
    return b


def simulate_bench(path_or_dict, base_dir=None, output_path=None, return_scores=False):
    """Devuelve la intensidad (n,n) en el detector; con return_scores=True
    devuelve (I, {label: energia_en_region})."""
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
    for ly in b.get("layers", []):
        u = angular_spectrum(u, px, lam, ly["z_m"] - z)
        z = ly["z_m"]
        h = np.load(rp(ly["height_npy"]))
        if h.shape != (n, n):
            raise ValueError(f"{ly.get('name', '?')}: height shape {h.shape} != ({n},{n})")
        h = h.astype(float) * ly["height_max_m"]
        u = u * phase_element(h, lam, ly["n_material"], ly["n_medium"], ly.get("kappa", 0.0))
    u = angular_spectrum(u, px, lam, b["detector"]["z_m"] - z)
    I = np.abs(u) ** 2
    if output_path:
        np.save(output_path, I)
    if return_scores:
        regs = b["detector"].get("regions", [])
        masks = detector_masks(n, px, regs)
        scores = {r.get("label", i): float((masks[i] * I).sum()) for i, r in enumerate(regs)}
        return I, scores
    return I
