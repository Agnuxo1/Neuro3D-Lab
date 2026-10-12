"""Two-dimensional vector Maxwell probe for a finite dielectric phase strip.

This requires the genuine NanoComp/Meep distribution (pymeep). It is a local
component-validation experiment, not a simulation of the whole three-dimensional
D2NN. Both polarization and numerical convergence must be reported separately.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from .evidence import write_json_new


def solve(resolution, component_name, material_index, with_strip,
          after_sources=200, pml=1.0, cell_x=10.0, cell_y=8.0, angle_degrees=0.0, dimensions=2):
    """Compute a monochromatic DFT slice using a pulsed source and exterior PML."""
    import meep as mp
    component = mp.Ez if component_name == "Ez" else mp.Hz
    geometry = []
    if with_strip:
        thickness_z = mp.inf if dimensions == 2 else 0.5333333333
        # Coordinates are measured in vacuum wavelengths; propagation is along +x.
        geometry = [mp.Block(center=mp.Vector3(-0.5, -0.2666666667),
                             size=mp.Vector3(0.5, 0.5333333333, thickness_z),
                             material=mp.Medium(index=material_index)),
                    mp.Block(center=mp.Vector3(-0.25, 0.2666666667),
                             size=mp.Vector3(1.0, 0.5333333333, thickness_z),
                             material=mp.Medium(index=material_index))]
    sources = [mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.15), component=component,
                         center=mp.Vector3(-3.0), size=mp.Vector3(0, 4.0, 4.0 if dimensions == 3 else 0),
                         amp_func=lambda r: np.exp(-(r.y/1.0)**2-(r.z/1.0)**2)*
                         np.exp(2j*np.pi*np.sin(np.deg2rad(angle_degrees))*r.y))]
    sim = mp.Simulation(cell_size=mp.Vector3(cell_x, cell_y, cell_y if dimensions == 3 else 0), geometry=geometry, sources=sources,
                        boundary_layers=[mp.PML(pml)], resolution=resolution,
                        dimensions=dimensions, force_complex_fields=True)
    monitor = sim.add_dft_fields([component], 1.0, 0, 1,
                                 center=mp.Vector3(2.0), size=mp.Vector3(0, 4.0, 4.0 if dimensions == 3 else 0))
    sim.run(until_after_sources=after_sources)
    field = np.asarray(sim.get_dft_array(monitor, component, 0))
    metadata = sim.get_array_metadata(dft_cell=monitor)
    coordinates = dict(y=np.asarray(metadata[1]), z=np.asarray(metadata[2]), weights=np.asarray(metadata[3]))
    sim.reset_meep()
    return field, coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--component", choices=["Ez", "Hz"], required=True)
    parser.add_argument("--index", type=float, default=1.5)
    parser.add_argument("--after-sources", type=float, default=200.0)
    parser.add_argument("--pml", type=float, default=1.0)
    parser.add_argument("--cell-x", type=float, default=10.0)
    parser.add_argument("--cell-y", type=float, default=8.0)
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--dimensions", type=int, choices=[2, 3], default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parameters = [args.index, args.after_sources, args.pml, args.cell_x, args.cell_y, args.angle]
    if (args.resolution < 8 or not np.isfinite(parameters).all() or args.index < 1 or
            args.after_sources <= 0 or args.pml <= 0 or args.cell_x <= 2*args.pml+6 or
            args.cell_y <= 2*args.pml+4 or abs(args.angle) >= 60):
        parser.error("invalid resolution, material, pulse, PML, domain, or incidence angle")
    import meep as mp
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("probe evidence already exists")
    settings = dict(after_sources=args.after_sources, pml=args.pml,
                    cell_x=args.cell_x, cell_y=args.cell_y, angle_degrees=args.angle, dimensions=args.dimensions)
    incident, y = solve(args.resolution, args.component, args.index, False, **settings)
    transmitted, y2 = solve(args.resolution, args.component, args.index, True, **settings)
    if any(not np.array_equal(y[key], y2[key]) for key in y):
        raise RuntimeError("air and material monitor grids differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, air_field=incident, material_field=transmitted, **y)
    write_json_new(args.output.with_suffix(".json"), dict(resolution=args.resolution,
            component=args.component, material_index=args.index, meep_version=mp.__version__,
            settings=settings, vacuum_wavelength_units=True,
            scope=f"{args.dimensions}D finite local element; PML/time/grid convergence not yet demonstrated"))


if __name__ == "__main__":
    main()
