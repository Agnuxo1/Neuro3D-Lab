"""Gate T0 for CONTRATO-008: torch batched propagation against optlab at N = 128.

CONTRATO-009 verified the torch path against optlab at N = 32 only. This gate repeats the same
comparison at the N = 128 grid, the one used by CONTRATO-003 and CONTRATO-008, with the same
thresholds as CONTRATO-009. No training, no TEST data.
Usage: python -B experiments/check_torch_vs_optlab_n128.py
Output: experiments/results/check_torch_vs_optlab_n128.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import run_d2nn_digits as R  # noqa: E402  (importing does not run main())
from optlab.d2nn import DiffractiveNetwork  # noqa: E402

torch.set_default_dtype(torch.float64)
N = 128
DX = 0.4e-3
LAM = 0.75e-3
DIST = [18e-3] * 4
L = len(DIST) - 1
K_WAVE = 2 * math.pi / LAM
TOL = {"I_rel": 1e-9, "score_rel": 1e-9, "loss_abs": 1e-10, "grad_rel": 1e-8}
BETA = 100.0


def transfer(d):
    f = torch.fft.fftfreq(N, d=DX)
    fx, fy = torch.meshgrid(f, f, indexing="xy")
    kx = 2 * math.pi * fx
    ky = 2 * math.pi * fy
    arg = K_WAVE**2 - kx**2 - ky**2
    prop = arg > 0
    kz = torch.sqrt(torch.where(prop, arg, torch.zeros_like(arg)))
    return torch.where(prop, torch.exp(1j * d * kz), torch.zeros(N, N, dtype=torch.complex128))


def torch_run(u0_np, y, phases_np, masks_t, H):
    ph = torch.tensor(phases_np, requires_grad=True)
    a = torch.from_numpy(u0_np)
    for k in range(L):
        a = torch.fft.ifft2(H[k] * torch.fft.fft2(a)) * torch.exp(1j * ph[k])
    f = torch.fft.ifft2(H[L] * torch.fft.fft2(a))
    I = f.abs() ** 2
    scores = BETA * torch.einsum("kij,ij->k", masks_t, I)
    loss = -torch.log_softmax(scores, dim=0)[y]
    loss.backward()
    return I.detach().numpy(), scores.detach().numpy(), float(loss.detach()), ph.grad.detach().numpy()


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    masks = R.build_masks()
    rng = np.random.default_rng(3)
    amp = np.zeros((N, N))
    amp[32:96, 32:96] = rng.uniform(0.0, 1.0, size=(64, 64))
    amp /= np.linalg.norm(amp)
    u0 = amp.astype(complex)
    phases = np.random.default_rng(0).uniform(0.0, 2 * math.pi, size=(L, N, N))
    label = 3

    dn = DiffractiveNetwork(n=N, pixel_m=DX, wavelength_m=LAM, distances_m=DIST,
                            detector_masks=masks, scale=BETA, seed=0)
    dn.phases = phases.copy()
    ref = dn.forward(u0)
    loss_ref, grad_ref = dn.loss_and_grad(u0, label)

    H = [transfer(d) for d in DIST]
    I_t, s_t, loss_t, g_t = torch_run(u0, label, phases, torch.from_numpy(masks), H)

    vals = {
        "I_rel": rel(I_t, ref["I"]),
        "score_rel": float(np.max(np.abs(s_t - ref["scores"])) / np.max(np.abs(ref["scores"]))),
        "loss_abs": abs(loss_t - loss_ref),
        "grad_rel": rel(g_t, grad_ref),
    }
    passed = {k: bool(vals[k] <= TOL[k]) for k in vals}
    res = {
        "contract": "CONTRATO-008 gate T0 (N = 128 torch versus optlab)",
        "N": N, "beta": BETA, "label": label, "tolerances": TOL,
        "values": vals, "pass": passed,
        "verdict": "PASS" if all(passed.values()) else "FAIL",
    }
    out = ROOT / "experiments" / "results" / "check_torch_vs_optlab_n128.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("values", "pass", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
