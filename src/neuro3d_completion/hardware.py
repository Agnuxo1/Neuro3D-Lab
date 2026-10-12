"""Analyze admitted calibrated measurements against fixed scalar predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from .evidence import digest_file, record, wilson


def compare(measured_path, simulation_path):
    """Compare detector power without fitting a scale to the evaluation samples.

    Measured NPZ: power_w(repeats,samples,regions), dark_w with the same shape,
    input_power_w(samples), labels(samples), sample_ids(samples), unit='W'.
    Simulation NPZ: region_energy(samples,regions), sample_ids(samples).
    region_energy is a dimensionless fraction of admitted input energy.
    """
    with np.load(measured_path, allow_pickle=False) as data:
        measured, dark = data["power_w"], data["dark_w"]
        input_power, y, ids = data["input_power_w"], data["labels"], data["sample_ids"]
        if str(data["unit"].item()) != "W":
            raise ValueError("measurement powers must be calibrated in watts")
    with np.load(simulation_path, allow_pickle=False) as data:
        predicted, sim_ids = data["region_energy"], data["sample_ids"]
    if (measured.ndim != 3 or measured.shape != dark.shape or measured.shape[0] < 2 or
            predicted.shape != measured.shape[1:] or input_power.shape != y.shape or
            len(y) != measured.shape[1] or ids.shape != y.shape or not np.array_equal(ids, sim_ids)):
        raise ValueError("unaligned measured and simulated samples or insufficient repeats")
    if (not np.issubdtype(y.dtype, np.integer) or np.any(y < 0) or
            np.any(y >= predicted.shape[1]) or len(np.unique(ids)) != len(ids)):
        raise ValueError("invalid labels or duplicate measured sample IDs")
    if (not np.isfinite(measured).all() or not np.isfinite(dark).all() or
            not np.isfinite(input_power).all() or not np.isfinite(predicted).all() or
            np.any(input_power <= 0) or np.any(predicted < 0)):
        raise ValueError("invalid measured powers or simulation energy")
    corrected = (measured-dark)/input_power[None, :, None]
    mean, sd = corrected.mean(axis=0), corrected.std(axis=0, ddof=1)
    denominator = np.linalg.norm(predicted)
    if denominator == 0:
        raise ValueError("zero simulated detector power")
    residual = mean-predicted
    labels = mean.argmax(axis=1)
    accuracy = float(np.mean(labels == y))
    correlations = []
    for region in range(predicted.shape[1]):
        a, b = mean[:, region], predicted[:, region]
        correlations.append(None if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1]))
    return dict(relative_L2=float(np.linalg.norm(residual)/denominator),
                absolute_RMSE=float(np.sqrt(np.mean(residual**2))),
                detector_correlations=correlations, repeat_sd_mean=float(sd.mean()),
                negative_dark_corrected_fraction=float(np.mean(corrected < 0)),
                accuracy=accuracy, wilson_95=wilson(int(np.sum(labels == y)), len(y)),
                samples=len(y), repeats=measured.shape[0],
                measured_sha256=digest_file(measured_path), simulation_sha256=digest_file(simulation_path),
                scope="calibrated supplied samples; no fitted gain and no automatic hardware-success verdict")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record(args.output, "calibrated hardware comparison", {}, compare(args.measured, args.simulation))
    print(args.output)


if __name__ == "__main__":
    main()
