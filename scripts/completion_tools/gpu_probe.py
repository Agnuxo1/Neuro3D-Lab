"""CUDA readiness probe for the completion protocol (guide sections 5.2, 13-E9 and 19).

1. Deterministic mode: run the NumPy/Torch equivalence gate and one training step of every
   optical and digital model on CUDA with torch.use_deterministic_algorithms(True); record any
   operation that has no deterministic implementation instead of hiding it.
2. Throughput: seconds per training sample (forward + backward + Adam) for each (padding,
   refinement) pair, complex128 reference precision, with synchronised timing.
The record is written once (no overwrite). It contains measurements only, no TEST data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from neuro3d_completion.benchmark import equivalence_gate
from neuro3d_completion.budget import measure
from neuro3d_completion.core import Optics
from neuro3d_completion.evidence import record
from neuro3d_completion.torch_backend import digital_model


def deterministic_checks(optics):
    """Equivalence gate plus one deterministic CUDA step of every model family."""
    out = {}
    try:
        out["equivalence_gate_cuda"] = equivalence_gate(optics, "cuda")
    except Exception as exc:  # recorded, never suppressed
        out["equivalence_gate_cuda"] = dict(error=repr(exc), trace=traceback.format_exc()[-800:])
    for name in ("linear", "mlp12", "mlp128", "cnn", "quadratic12"):
        try:
            torch.manual_seed(0)
            model = digital_model(name).to("cuda")
            x = torch.rand(8, 1, 64, 64, dtype=torch.float64, device="cuda")
            y = torch.arange(8, device="cuda") % 10
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            out[f"step_{name}"] = dict(loss=float(loss), ok=True)
        except Exception as exc:
            out[f"step_{name}"] = dict(ok=False, error=repr(exc))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, nargs="+", default=[2, 1, 4, 1, 8, 1, 2, 2, 4, 2, 2, 4, 4, 4, 2, 8],
                        help="flat padding/refinement pairs")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--large-batch", type=int, default=4, help="batch used when m >= 4096")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = dict(n=128, pixel_m=4e-4, wavelength_m=7.5e-4, distances_m=(0.018,)*4, boundary="aperture")
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                  deterministic=deterministic_checks(Optics(**base, padding=2, refinement=1)), throughput=[])
    for padding, refinement in zip(args.pairs[0::2], args.pairs[1::2]):
        optics = Optics(**base, padding=padding, refinement=refinement)
        batch = args.large_batch if optics.m >= 4096 else args.batch
        try:
            row = measure(optics, "cuda", batch=batch, steps=5, warmup=2)
        except Exception as exc:
            row = dict(m=optics.m, batch=batch, error=repr(exc))
        row.update(padding=padding, refinement=refinement)
        result["throughput"].append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()
    record(args.output, "CUDA readiness and throughput probe", base, result)
    print(args.output)


if __name__ == "__main__":
    main()
