"""Accuracy versus optical energy under detection noise, from frozen detector energies.

Two explicitly different noise models are provided; neither is a hardware measurement:

* photon counting: counts_k ~ Poisson(eta * N * E_k + dark) + Normal(0, read_noise),
  optional saturation and ADC quantisation. Appropriate where shot noise can dominate
  (visible/near-infrared photon-counting detectors).
* NEP: power_k = P_in * E_k + Normal(0, NEP * sqrt(bandwidth)), bandwidth = 1/(2 tau).
  Appropriate for room-temperature THz detection, which is normally NEP-limited.

E_k is the fraction of input energy reaching detector k (simulation 'region_energy',
dimensionless). At lambda = 0.75 mm, h*nu = 2.65e-22 J while k_B*T(300 K) = 4.14e-21 J,
so the thermal photon occupation is about 15: a shot-noise-limited photon budget is not a
realistic room-temperature description at 0.4 THz. Use the NEP model there, with the
detector's own NEP definition and bandwidth convention from its datasheet.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

from .evidence import digest_file, record, wilson
from .guards import frozen_optics, refuse_reserved_split

PLANCK = 6.62607015e-34
LIGHT_SPEED = 299792458.0
BOLTZMANN = 1.380649e-23


def photon_energy(wavelength_m):
    """Energy of one photon in joules."""
    if not wavelength_m > 0:
        raise ValueError("wavelength must be positive")
    return PLANCK*LIGHT_SPEED/wavelength_m


def thermal_occupation(wavelength_m, temperature_k):
    """Mean Bose-Einstein occupation 1/(exp(h nu / k T) - 1)."""
    if not temperature_k > 0:
        raise ValueError("temperature must be positive")
    return 1.0/math.expm1(photon_energy(wavelength_m)/(BOLTZMANN*temperature_k))


def _validate(region_energy, labels):
    e, y = np.asarray(region_energy, float), np.asarray(labels)
    if (e.ndim != 2 or e.shape[0] == 0 or e.shape[1] < 2 or y.shape != (e.shape[0],) or
            not np.isfinite(e).all() or np.any(e < 0) or np.any(e.sum(axis=1) > 1 + 1e-9)):
        raise ValueError("region_energy must be finite fractions (samples, classes) with total <= 1")
    if not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or np.any(y >= e.shape[1]):
        raise ValueError("labels must be valid class indices")
    return e, y


def counting_accuracy(region_energy, labels, photons, efficiency=1.0, dark_counts=0.0, read_noise=0.0,
                      full_well=None, adc_bits=None, repeats=20, seed=0):
    """Monte Carlo accuracy at a fixed number of input photons per inference."""
    e, y = _validate(region_energy, labels)
    if photons < 0 or not 0 < efficiency <= 1 or dark_counts < 0 or read_noise < 0 or repeats < 1:
        raise ValueError("invalid detection parameters")
    rng = np.random.default_rng(seed)
    mean = efficiency*photons*e + dark_counts
    correct = 0
    for _ in range(repeats):
        counts = rng.poisson(mean).astype(float)
        if read_noise:
            counts += rng.normal(0, read_noise, counts.shape)
        if full_well is not None:
            counts = np.clip(counts, 0, full_well)
            if adc_bits is not None:
                step = full_well/(2**adc_bits - 1)
                counts = np.round(counts/step)*step
        noise = rng.uniform(0, 1e-9, counts.shape)  # random tie-breaking, never systematic
        correct += int(np.sum((counts + noise).argmax(axis=1) == y))
    total = repeats*len(y)
    return dict(accuracy=correct/total, wilson_95=wilson(correct, total), photons=float(photons),
                repeats=repeats)


def nep_accuracy(region_energy, labels, input_power_w, nep_w_per_sqrt_hz, integration_s, repeats=20, seed=0):
    """Monte Carlo accuracy with additive Gaussian power noise sigma = NEP*sqrt(1/(2 tau))."""
    e, y = _validate(region_energy, labels)
    if input_power_w < 0 or nep_w_per_sqrt_hz < 0 or integration_s <= 0 or repeats < 1:
        raise ValueError("invalid NEP parameters")
    sigma = nep_w_per_sqrt_hz*math.sqrt(1/(2*integration_s))
    rng = np.random.default_rng(seed)
    correct = 0
    for _ in range(repeats):
        power = input_power_w*e + rng.normal(0, sigma, e.shape)
        correct += int(np.sum(power.argmax(axis=1) == y))
    total = repeats*len(y)
    return dict(accuracy=correct/total, wilson_95=wilson(correct, total), input_power_w=float(input_power_w),
                noise_sigma_w=sigma, optical_energy_j=float(input_power_w*integration_s), repeats=repeats)


def photon_curve(region_energy, labels, wavelength_m, photon_grid, **kwargs):
    """Accuracy over a photon grid plus the noiseless reference accuracy and energies."""
    e, y = _validate(region_energy, labels)
    noiseless = float(np.mean(e.argmax(axis=1) == y))
    rows = []
    for photons in photon_grid:
        row = counting_accuracy(e, y, photons, **kwargs)
        row["optical_energy_j"] = float(photons*photon_energy(wavelength_m))
        rows.append(row)
    return dict(noiseless_accuracy=noiseless, rows=rows, wavelength_m=wavelength_m,
                photon_energy_j=photon_energy(wavelength_m),
                thermal_occupation_300K=thermal_occupation(wavelength_m, 300.0))


def export_region_energy(phases_path, split_path, output, limit=None, batch=16):
    """Write region_energy (fraction of unit input energy per detector), labels and sample_ids.

    Uses the optics stored with the frozen phases by benchmark.final. The NPZ is the
    'simulation' input of hardware.py and of photon_curve. Refuses TEST splits: the
    hardware evaluation set must be chosen and registered separately.
    """
    from .core import DiffractiveNetwork, detector_masks
    from .data import encode
    refuse_reserved_split(split_path)
    # architecture.json (written by benchmark.final) is authoritative, including filter_mode.
    optics, _ = frozen_optics(phases_path)
    with np.load(phases_path, allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    with np.load(split_path, allow_pickle=False) as data:
        images, labels, ids = data["images"][:limit], data["labels"][:limit], data["sample_ids"][:limit]
    x = encode(images)
    off = (optics.n - x.shape[-1])//2
    x = np.pad(x, ((0, 0), (off, off), (off, off)))
    model = DiffractiveNetwork(optics, detector_masks(optics.n, optics.pixel_m), beta=1.0)
    model.phases = phases
    energy = np.concatenate([model.forward(x[i:i+batch])["scores"] for i in range(0, len(x), batch)])
    with Path(output).open("xb") as stream:
        np.savez_compressed(stream, region_energy=energy, labels=labels, sample_ids=ids)
    return energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-phases", type=Path, help="frozen phases.npz; with --export-split writes --output")
    parser.add_argument("--export-split", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--simulation", type=Path,
                        help="NPZ with region_energy (samples, classes) and labels (samples)")
    parser.add_argument("--wavelength", type=float, default=7.5e-4)
    parser.add_argument("--photons", type=float, nargs="+", default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    parser.add_argument("--efficiency", type=float, default=1.0)
    parser.add_argument("--dark", type=float, default=0.0)
    parser.add_argument("--read-noise", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.export_phases:
        if args.export_split is None:
            parser.error("--export-split is required with --export-phases")
        export_region_energy(args.export_phases, args.export_split, args.output, args.limit)
        print(args.output)
        return
    if args.simulation is None:
        parser.error("--simulation is required for an accuracy curve")
    with np.load(args.simulation, allow_pickle=False) as data:
        energy, labels = data["region_energy"], data["labels"]
    result = photon_curve(energy, labels, args.wavelength, args.photons, efficiency=args.efficiency,
                          dark_counts=args.dark, read_noise=args.read_noise, repeats=args.repeats)
    result["simulation_sha256"] = digest_file(args.simulation)
    result["scope"] = "idealised photon-counting model on frozen simulated energies; not a measurement"
    record(args.output, "photon-noise accuracy curve", vars(args) | {"simulation": str(args.simulation),
           "output": str(args.output)}, result)
    print(args.output)


if __name__ == "__main__":
    main()
