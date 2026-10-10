"""CONTRATO-009: independent PyTorch replication of optlab's D2NN forward pass and analytic gradient.

Pre-registered in docs/CONTRACTS/CONTRATO-009-pytorch-replication.md. The thresholds are fixed there.
The replication implements the propagator and the gradient from their definitions (autograd, own FFT
calls). It takes only the phases and the reference values from optlab.
Usage: python -B experiments/replicate_d2nn_pytorch.py
Output: experiments/results/replicate_d2nn_pytorch.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optlab.d2nn import DiffractiveNetwork  # noqa: E402

torch.set_default_dtype(torch.float64)
N = 32
PIX = 0.4e-3
LAM = 0.75e-3
DIST = [18e-3, 18e-3, 18e-3]
L = len(DIST) - 1
K_DET = 10
K_WAVE = 2 * math.pi / LAM
LABELS = [0, 3, 5, 9]
TOL = {"T1": 1e-9, "T2": 1e-9, "T3": 1e-10, "T4": 1e-8, "T5": 1e-5}


def detector_masks():
    m = np.zeros((K_DET, N, N))
    rows = [6, 22]
    cols = [1, 7, 13, 19, 25]
    for k in range(K_DET):
        r, c = rows[k // 5], cols[k % 5]
        m[k, r:r + 4, c:c + 4] = 1.0
    return m


def inputs():
    rng = np.random.default_rng(1)
    amp = rng.uniform(0.2, 1.0, size=(len(LABELS), N, N))
    th = rng.uniform(0.0, 2 * math.pi, size=(len(LABELS), N, N))
    return amp * np.exp(1j * th)


def transfer(d):
    """H = exp(i d kz) for propagating components, zero for evanescent ones (own definition)."""
    f = torch.fft.fftfreq(N, d=PIX)
    fx, fy = torch.meshgrid(f, f, indexing="xy")
    kx = 2 * math.pi * fx
    ky = 2 * math.pi * fy
    arg = K_WAVE**2 - kx**2 - ky**2
    prop = arg > 0
    kz = torch.sqrt(torch.where(prop, arg, torch.zeros_like(arg)))
    return torch.where(prop, torch.exp(1j * d * kz), torch.zeros(N, N, dtype=torch.complex128))


def torch_forward_and_grad(u0_np, y, phases_np, masks_t, H):
    ph = torch.tensor(phases_np, requires_grad=True)
    a = torch.from_numpy(u0_np)
    for k in range(L):
        a = torch.fft.ifft2(H[k] * torch.fft.fft2(a)) * torch.exp(1j * ph[k])
    f = torch.fft.ifft2(H[L] * torch.fft.fft2(a))
    intensity = f.abs() ** 2
    scores = torch.einsum("kij,ij->k", masks_t, intensity)
    loss = -torch.log_softmax(scores, dim=0)[y]
    loss.backward()
    return (intensity.detach().numpy(), scores.detach().numpy(),
            float(loss.detach()), ph.grad.detach().numpy())


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    masks = detector_masks()
    masks_t = torch.from_numpy(masks)
    dn = DiffractiveNetwork(n=N, pixel_m=PIX, wavelength_m=LAM, distances_m=DIST,
                            detector_masks=masks, scale=1.0, seed=0)
    phases = dn.phases.copy()
    H = [transfer(d) for d in DIST]
    U0 = inputs()

    t1, t2, t3, t4 = [], [], [], []
    per_input = []
    for i, y in enumerate(LABELS):
        I_t, s_t, loss_t, g_t = torch_forward_and_grad(U0[i], y, phases, masks_t, H[:L + 1])
        ref = dn.forward(U0[i])
        loss_ref, g_ref = dn.loss_and_grad(U0[i], y)
        t1.append(rel(I_t, ref["I"]))
        t2.append(float(np.max(np.abs(s_t - ref["scores"])) / np.max(np.abs(ref["scores"]))))
        t3.append(abs(loss_t - loss_ref))
        t4.append(rel(g_t, g_ref))
        per_input.append({"label": y, "loss_optlab": loss_ref, "loss_torch": loss_t,
                          "grad_norm_optlab": float(np.linalg.norm(g_ref))})

    # T5: optlab analytic gradient against central differences on 5 entries, each input
    rng = np.random.default_rng(2)
    idxs = [(int(rng.integers(L)), int(rng.integers(N)), int(rng.integers(N))) for _ in range(5)]
    h = 1e-6
    t5 = []
    for i, y in enumerate(LABELS):
        _, g_ref = dn.loss_and_grad(U0[i], y)
        for idx in idxs:
            base = dn.phases[idx]
            dn.phases[idx] = base + h
            lp, _ = dn.loss_and_grad(U0[i], y)
            dn.phases[idx] = base - h
            lm, _ = dn.loss_and_grad(U0[i], y)
            dn.phases[idx] = base
            fd = (lp - lm) / (2 * h)
            an = g_ref[idx]
            t5.append(abs(fd - an) / max(abs(an), 1e-8))
    assert np.allclose(dn.phases, phases), "phases were not restored after finite differences"

    vals = {"T1": max(t1), "T2": max(t2), "T3": max(t3), "T4": max(t4), "T5": max(t5)}
    passed = {k: bool(vals[k] <= TOL[k]) for k in vals}
    res = {
        "contract": "CONTRATO-009 (pre-registered)",
        "config": {"n": N, "pixel_m": PIX, "wavelength_m": LAM, "distances_m": DIST,
                   "detectors": K_DET, "seed_phases": 0, "labels": LABELS},
        "values": vals,
        "tolerances": TOL,
        "pass": passed,
        "verdict": "PASS" if all(passed.values()) else "FAIL",
        "per_input": per_input,
        "fd_entries": [list(x) for x in idxs],
    }
    out = ROOT / "experiments" / "results" / "replicate_d2nn_pytorch.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("values", "pass", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
