"""Measure training throughput on the target device and estimate the protocol's compute budget.

Run this before freezing a protocol: padding and refinement multiply the FFT size
m = n*r*p, and a configuration that is numerically adequate may be computationally
infeasible. The estimate counts optical training passes; digital baselines are cheap
by comparison and are reported separately only if measured. GeForce RTX 3090 (GA102)
runs FP64 at 1/64 of its FP32 rate (NVIDIA GA102 whitepaper), so complex128 FFTs are
mostly memory-bound there; measure instead of assuming.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from torch.nn import functional as F

from .core import Optics, DiffractiveNetwork, detector_masks
from .torch_backend import TorchD2NN

try:  # resource is POSIX-only; Windows falls back to psutil when installed
    import resource
except ImportError:
    resource = None


def _cpu_peak_mib():
    """Process peak resident memory in MiB, or None when no portable source exists."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    try:
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss)/2**20
    except ImportError:
        return None


def measure(optics, device="cpu", batch=16, steps=3, warmup=1, dtype=torch.float64, seed=0):
    """Seconds per training sample (forward + backward + Adam) and peak memory."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if dtype != torch.float64:
        raise ValueError("only the float64/complex128 reference precision is measured here; "
                         "reduced precision is a separate validated variant")
    model = TorchD2NN(DiffractiveNetwork(optics, detector_masks(optics.n, optics.pixel_m), seed=seed)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(rng.uniform(0, 1, (batch, optics.n, optics.n)), dtype=torch.float64, device=device)
    y = torch.as_tensor(rng.integers(0, 10, batch), device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rss_before = _cpu_peak_mib()
    timings = []
    for step in range(warmup + steps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        if step >= warmup:
            timings.append(time.perf_counter() - t0)
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated()/2**20
    else:
        after = _cpu_peak_mib()
        peak = None if after is None or rss_before is None else after - rss_before
    return dict(m=optics.m, batch=batch, device=device, seconds_per_sample=float(np.median(timings))/batch,
                peak_memory_mib=None if peak is None else float(peak), memory_note=("cuda: allocator peak; cpu: growth of the process peak RSS (ru_maxrss), "
                             "0 if an earlier step already peaked higher: measure in a fresh process"),
                torch=torch.__version__, platform=platform.platform(),
                gpu=torch.cuda.get_device_name() if device == "cuda" else None)


def estimate(protocol, seconds_per_sample, devices=1, eval_cost_ratio=0.35, train_size=50000,
             dev_size=10000, test_size=10000):
    """Optical-pass hours for DEV selection and final runs under the protocol's grids."""
    epochs, seeds = protocol["epochs"], len(protocol["seeds"])
    grid = len(protocol.get("beta_grid", [100, 200, 400]))*len(protocol.get("optical_lr_grid", [0.02, 0.05]))
    per_run_train = epochs*train_size*seconds_per_sample
    per_run_eval = (epochs*dev_size + test_size)*seconds_per_sample*eval_cost_ratio
    selection = grid*(per_run_train + epochs*dev_size*seconds_per_sample*eval_cost_ratio)
    trained_final = 2*seeds*(per_run_train + per_run_eval)          # d2nn and permuted-label control
    untrained_final = 2*seeds*test_size*seconds_per_sample*eval_cost_ratio  # untrained and no-layer controls
    total = selection + trained_final + untrained_final
    return dict(selection_hours=selection/3600, final_hours=(trained_final + untrained_final)/3600,
                total_device_hours=total/3600,
                wall_hours_parallel=(selection/min(devices, grid) + (trained_final + untrained_final)/devices)/3600,
                devices=devices, assumptions=dict(eval_cost_ratio=eval_cost_ratio, train_size=train_size,
                                                  dev_size=dev_size, test_size=test_size,
                                                  note="optical model only, one dataset; double for two datasets"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--padding", type=int, help="override the protocol padding for a what-if run")
    parser.add_argument("--refinement", type=int, help="override the protocol refinement for a what-if run")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    settings = dict(protocol["optics"])
    if args.padding:
        settings["padding"] = args.padding
    if args.refinement:
        settings["refinement"] = args.refinement
    optics = Optics(**settings)
    measured = measure(optics, args.device, args.batch)
    print(json.dumps(dict(measurement=measured, estimate=estimate(protocol, measured["seconds_per_sample"],
                                                                   args.devices)), indent=2))


if __name__ == "__main__":
    main()
