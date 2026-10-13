"""Second refinement triplet (4-8-16) for the historical masks, on the GPU (guide §16, pending item).

Scores of the historical seed-0 phases for the two smooth diagnostic inputs of
`validation.pixel_convergence` (Gaussian and tilted), padding 2, refinement 4, 8 and 16, computed with
the lean GPU constructor of `fidelity_study.py` (identity with TorchD2NN tested). Reports the relative
L2 change of the scores between levels and the observed order of the 4-8-16 triplet, next to the
2-4-8 triplet of CONTRATO-012. Scores only: fields at m = 4096 are not stored.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity_study import lean_model  # noqa: E402
from neuro3d_completion.convergence import observed_order_from_differences  # noqa: E402
from neuro3d_completion.core import Optics  # noqa: E402
from neuro3d_completion.evidence import digest_file, record  # noqa: E402
from neuro3d_completion.validation import relative, source_field  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", type=Path, default=Path("experiments/results/d2nn_seed0_phases.npz"))
    parser.add_argument("--refinements", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    with np.load(args.phases, allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    base = Optics(n=128, padding=2, refinement=1, boundary="aperture")
    inputs = np.asarray([source_field(base.n, base.pixel_m, base.wavelength_m, kind) for kind in ("gaussian", "tilted")])
    inputs /= np.linalg.norm(inputs, axis=(1, 2))[:, None, None]
    scores = {}
    for r in args.refinements:
        o = Optics(n=128, padding=2, refinement=r, boundary="aperture")
        model = lean_model(phases, o, 100.0, args.device)
        with torch.no_grad():
            scores[r] = np.concatenate([model(torch.as_tensor(inputs[i:i+1], device=args.device)).cpu().numpy()
                                        for i in range(len(inputs))])
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    levels = sorted(scores)
    steps = {f"refinement_{a}_vs_{b}": relative(scores[a], scores[b]) for a, b in zip(levels[:-1], levels[1:])}
    result = dict(scores={str(r): s.tolist() for r, s in scores.items()}, score_relative_L2=steps,
                  phases_sha256=digest_file(args.phases), base_padding=2,
                  scope="historical seed-0 masks, two smooth diagnostic inputs, scores only")
    if len(levels) == 3:
        a, b = steps[f"refinement_{levels[0]}_vs_{levels[1]}"], steps[f"refinement_{levels[1]}_vs_{levels[2]}"]
        result["observed_order_triplet"] = observed_order_from_differences(a, b, levels[1]/levels[0])
    record(args.output, "refinement 4-8-16 of the historical masks (scores)", dict(refinements=levels), result)
    print(json.dumps({k: result[k] for k in ("score_relative_L2", "observed_order_triplet") if k in result}, indent=1))


if __name__ == "__main__":
    main()
