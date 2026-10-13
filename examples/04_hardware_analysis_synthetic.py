"""Example 4: the calibrated-measurement analysis on SYNTHETIC numbers (no hardware data exists).

Run: python examples/04_hardware_analysis_synthetic.py --output <new folder>
It writes a simulated detector-energy file and a fake "measurement" built from it with a known
dark offset and noise, then runs hardware.compare. The numbers mean nothing physically; the
example only shows the required formats (power in W, dark frames, repeats, sample IDs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from neuro3d_completion.hardware import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    samples, regions, repeats = 20, 10, 3
    energy = rng.dirichlet(np.ones(regions), size=samples)*0.6
    labels = energy.argmax(axis=1)
    input_power = np.full(samples, 1e-3)
    dark = np.full((repeats, samples, regions), 2e-7)
    power = energy[None]*input_power[None, :, None] + dark + rng.normal(0, 1e-7, (repeats, samples, regions))
    np.savez(args.output/"simulation.npz", region_energy=energy, sample_ids=np.arange(samples))
    np.savez(args.output/"measured_SYNTHETIC.npz", power_w=power, dark_w=dark, input_power_w=input_power,
             labels=labels, sample_ids=np.arange(samples), unit="W")
    result = compare(args.output/"measured_SYNTHETIC.npz", args.output/"simulation.npz")
    print(json.dumps({k: result[k] for k in ("relative_L2", "accuracy", "repeats", "samples")}, indent=1))
    print("SYNTHETIC DATA: not a measurement")


if __name__ == "__main__":
    main()
