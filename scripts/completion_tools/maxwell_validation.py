"""CONTRATO-015: 2D vector (Maxwell, Meep FDTD) checks of the scalar thin-element model.

Runs inside the Meep environment (conda-forge `pymeep`; on this workstation WSL Ubuntu with the
environment on D:). Units: vacuum wavelength = 1. Propagation along +x; transverse coordinate y.
Three Meep runs per (component, resolution), all with the same cell, source and monitors:
  air      : nothing in the cell
  slab     : uniform glass slab n = 1.5, x in [-0.75, 0.25] (thickness 1), infinite in y
  element  : the two-pixel element of `neuro3d_completion.maxwell_probe` (pixels 0.533 wide,
             thickness 0.5 for y in [-0.533, 0] and 1.0 for y in [0, 0.533], front face x = -0.75)
Monitors (complex DFT at the source frequency): plane A at x = -1.5 (incident side) and plane B at
x = 2.0, both spanning |y| <= 3. Nothing is fitted, aligned or renormalised beyond dividing by the
air run of the same resolution where stated.

Checks recorded (thresholds registered in docs/CONTRACTS/CONTRATO-015-maxwell.md):
  V0 vacuum   : scalar ASM of the air field A -> B against the Meep air field at B (|y| <= 1.5)
  V1 slab     : Meep slab/air transmission at the centre against the Fabry-Perot formula
  V2 discret. : change between resolutions of the element field at B divided by ||air at B||
  V3 element  : scalar thin-element prediction (phase delay at the front face, optional Fresnel
                amplitude) against Meep at B: detector-bin powers and phases
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

N_GLASS = 1.5
FRONT = -0.75
PIXEL = 0.5333333333
X_A, X_B = -1.5, 2.0
HALF_SPAN = 3.0


def run_meep(resolution, component_name, geometry_kind, source_width, after_sources):
    import meep as mp
    component = mp.Ez if component_name == "Ez" else mp.Hz
    material = mp.Medium(index=N_GLASS)
    if geometry_kind == "air":
        geometry = []
    elif geometry_kind == "slab":
        geometry = [mp.Block(center=mp.Vector3(FRONT + 0.5, 0), size=mp.Vector3(1.0, mp.inf, mp.inf), material=material)]
    else:
        geometry = [mp.Block(center=mp.Vector3(FRONT + 0.25, -PIXEL/2), size=mp.Vector3(0.5, PIXEL, mp.inf),
                             material=material),
                    mp.Block(center=mp.Vector3(FRONT + 0.5, PIXEL/2), size=mp.Vector3(1.0, PIXEL, mp.inf),
                             material=material)]
    sources = [mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.15), component=component,
                         center=mp.Vector3(-3.0), size=mp.Vector3(0, 2*HALF_SPAN),
                         amp_func=lambda r: math.exp(-(r.y/source_width)**2))]
    sim = mp.Simulation(cell_size=mp.Vector3(10, 2*HALF_SPAN + 2), geometry=geometry, sources=sources,
                        boundary_layers=[mp.PML(1.0)], resolution=resolution, dimensions=2,
                        force_complex_fields=True)
    monitors = {name: sim.add_dft_fields([component], 1.0, 0, 1, center=mp.Vector3(x), size=mp.Vector3(0, 2*HALF_SPAN))
                for name, x in (("A", X_A), ("B", X_B))}
    sim.run(until_after_sources=after_sources)
    out = {}
    for name, monitor in monitors.items():
        field = np.asarray(sim.get_dft_array(monitor, component, 0)).ravel()
        y = np.asarray(sim.get_array_metadata(dft_cell=monitor)[1]).ravel()
        out[name], out["y"] = field, y
    sim.reset_meep()
    return out


def asm_1d(field, y, distance, pad=8):
    """Scalar angular-spectrum propagation along x of a line field (lambda = 1); evanescent removed."""
    dy = float(np.mean(np.diff(y)))
    m = pad*len(field)
    padded = np.zeros(m, complex)
    start = (m - len(field))//2
    padded[start:start+len(field)] = field
    f = np.fft.fftfreq(m, d=dy)
    q2 = 1.0 - f**2
    h = np.where(q2 > 0, np.exp(2j*np.pi*distance*np.sqrt(np.maximum(q2, 0))), 0)
    return np.fft.ifft(h*np.fft.fft(padded))[start:start+len(field)]


def fabry_perot(n, thickness):
    """Normal-incidence slab transmission relative to the same thickness of air (time e^{-i w t})."""
    k = 2*np.pi
    r21, t12, t21 = (n - 1)/(n + 1), 2/(1 + n), 2*n/(1 + n)
    t = t12*t21*np.exp(1j*k*n*thickness)/(1 - r21**2*np.exp(2j*k*n*thickness))
    return t*np.exp(-1j*k*thickness)


def thin_element(air_a, y, fresnel):
    """Propagate the incident field to the front face, apply the thin phase delay, propagate to B."""
    height = np.where((y >= -PIXEL) & (y < 0), 0.5, np.where((y >= 0) & (y < PIXEL), 1.0, 0.0))
    t = np.exp(2j*np.pi*(N_GLASS - 1)*height)
    if fresnel:
        t = np.where(height > 0, t*(2/(1 + N_GLASS))*(2*N_GLASS/(1 + N_GLASS)), t)
    front = asm_1d(air_a, y, FRONT - X_A)
    return asm_1d(front*t, y, X_B - FRONT)


def detector_bins(y):
    edges = np.arange(-3, 4)*PIXEL
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


def compare_element(maxwell_b, scalar_b, air_b, y):
    norm = np.linalg.norm(air_b)
    bins = []
    for a, b in detector_bins(y):
        sel = (y >= a) & (y < b)
        pm, ps = float(np.sum(np.abs(maxwell_b[sel])**2)), float(np.sum(np.abs(scalar_b[sel])**2))
        bins.append(dict(y_from=a, y_to=b, maxwell_power=pm/norm**2, scalar_power=ps/norm**2,
                         relative_difference=abs(ps - pm)/pm if pm > 0 else None))
    peak = max(x["maxwell_power"] for x in bins)
    strong = [x for x in bins if x["maxwell_power"] >= 0.05*peak]
    amp = np.abs(scalar_b)
    mask = amp >= 0.3*amp.max()
    phase = np.angle(maxwell_b[mask]/scalar_b[mask])
    return dict(field_relative_L2=float(np.linalg.norm(maxwell_b - scalar_b)/np.linalg.norm(maxwell_b)),
                bins=bins, max_bin_power_difference=max(x["relative_difference"] for x in strong),
                max_abs_phase_difference_rad=float(np.max(np.abs(phase))),
                rms_phase_difference_rad=float(np.sqrt(np.mean(phase**2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--component", choices=["Ez", "Hz"], required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", required=True)
    parser.add_argument("--source-width", type=float, default=2.0)
    parser.add_argument("--after-sources", type=float, default=200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import meep as mp
    if args.output.exists():
        raise FileExistsError(args.output)
    runs, record = {}, dict(component=args.component, meep_version=mp.__version__, n_glass=N_GLASS,
                            source_width=args.source_width, after_sources=args.after_sources, levels={})
    for res in args.resolutions:
        runs[res] = {kind: run_meep(res, args.component, kind, args.source_width, args.after_sources)
                     for kind in ("air", "slab", "element")}
        air, slab, element, y = runs[res]["air"], runs[res]["slab"], runs[res]["element"], runs[res]["air"]["y"]
        centre = np.abs(y) <= 1.5
        vacuum = asm_1d(air["A"], y, X_B - X_A)
        mid = np.abs(y) <= 0.25
        t_meep = complex(np.mean(slab["B"][mid]/air["B"][mid]))
        t_fp = complex(fabry_perot(N_GLASS, 1.0))
        record["levels"][str(res)] = dict(
            V0_vacuum_relative_L2=float(np.linalg.norm(vacuum[centre] - air["B"][centre])/np.linalg.norm(air["B"][centre])),
            V1_slab=dict(meep=[t_meep.real, t_meep.imag], fabry_perot=[t_fp.real, t_fp.imag], abs_error=abs(t_meep - t_fp),
                         meep_power=abs(t_meep)**2, fabry_perot_power=abs(t_fp)**2),
            V3_phase_only=compare_element(element["B"], thin_element(air["A"], y, False), air["B"], y),
            V3_with_fresnel=compare_element(element["B"], thin_element(air["A"], y, True), air["B"], y))
    levels = sorted(args.resolutions)
    record["V2_discretisation"] = {}
    for lo, hi in zip(levels[:-1], levels[1:]):
        y_lo, y_hi = runs[lo]["air"]["y"], runs[hi]["air"]["y"]
        a = runs[lo]["element"]["B"]/np.linalg.norm(runs[lo]["air"]["B"])
        b_hi = runs[hi]["element"]["B"]/np.linalg.norm(runs[hi]["air"]["B"])
        b = np.interp(y_lo, y_hi, b_hi.real) + 1j*np.interp(y_lo, y_hi, b_hi.imag)
        b *= np.sqrt(len(y_hi)/len(y_lo))
        record["V2_discretisation"][f"{lo}_vs_{hi}"] = float(np.linalg.norm(a - b)/np.linalg.norm(b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    with args.output.with_suffix(".npz").open("xb") as stream:
        np.savez_compressed(stream, **{f"r{res}_{kind}_{plane}": runs[res][kind][plane] for res in runs
                                       for kind in ("air", "slab", "element") for plane in ("A", "B")},
                            **{f"r{res}_y": runs[res]["air"]["y"] for res in runs})
    print(json.dumps({k: v for k, v in record.items() if k != "levels"}, indent=1))


if __name__ == "__main__":
    main()
