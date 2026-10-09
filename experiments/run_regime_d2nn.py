"""CONTRACT-004 (docs/CONTRACTS/CONTRATO-004-regimen-d2nn.md): propagation in the D2NN sampling regime.

Runs T1 (analytic tilted plane waves), T2 (zero-padding convergence) and T3 (pixel convergence),
using the amended design. Thresholds are the ones fixed in the contract and are not changed here.

Usage: python experiments/run_regime_d2nn.py
Output: experiments/results/regime_d2nn_summary.json (small, versioned) and the full record in
$NEURO3D_LAB_OUT/results (default: <repo parent>/.cognition/neuro3d-lab/results).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlab.propagation import angular_spectrum  # noqa: E402

LAM = 0.75e-3          # wavelength [m]
D = 18e-3              # propagation distance [m]
DIGIT_SIZE = 25.6e-3   # physical width of the upsampled digit [m]
OUT_REPO = ROOT / "experiments" / "results"
OUT_LOCAL = Path(os.environ.get("NEURO3D_LAB_OUT")
                 or Path(__file__).resolve().parents[2] / ".cognition" / "neuro3d-lab") / "results"


def encode(img8, n, dx):
    """Digit (8x8, values in [0, 1]) upsampled to DIGIT_SIZE, unit energy, centred on an n x n grid."""
    pixels = round(DIGIT_SIZE / dx)
    scale = pixels // 8
    a = np.kron(img8, np.ones((scale, scale)))
    a = a / np.linalg.norm(a)
    pad = (n - pixels) // 2
    assert pixels + 2 * pad == n, "digit must be centred exactly"
    return np.pad(a, pad).astype(complex)


def intensity(u, dx):
    return np.abs(angular_spectrum(u, dx, LAM, D)) ** 2


def t1_tilted_plane_waves():
    n, dx = 128, 0.4e-3
    x = np.arange(n) * dx
    X = np.tile(x, (n, 1))
    errors = []
    for k in range(64):                       # grid frequencies below the Nyquist limit
        f = k / (n * dx)
        u0 = np.exp(1j * 2 * np.pi * f * X)
        u = angular_spectrum(u0, dx, LAM, D)
        exact = np.exp(1j * 2 * np.pi * (f * X + D * np.sqrt(1 / LAM**2 - f**2)))
        errors.append(float(np.linalg.norm(u - exact) / np.linalg.norm(exact)))
    max_k = int(np.argmax(errors))
    return {
        "k_range": [0, 63],
        "max_rel_error": max(errors),
        "argmax_k": max_k,
        "max_sin_theta": LAM * 63 / (n * dx),
        "per_k_rel_error": errors,
        "threshold": 1e-10,
        "verdict": "pass" if max(errors) <= 1e-10 else "fail",
    }


def t2_zero_padding(img8):
    dx = 0.4e-3
    i128 = intensity(encode(img8, 128, dx), dx)
    i256 = intensity(encode(img8, 256, dx), dx)
    centre = i256[64:192, 64:192]
    err = float(np.linalg.norm(i128 - centre) / np.linalg.norm(centre))
    return {
        "grids": {"coarse_N": 128, "padded_N": 256, "dx_m": dx},
        "validity_limits_m": {"N128": 128 * dx * dx / LAM, "N256": 256 * dx * dx / LAM},
        "rel_l2_intensity": err,
        "threshold": 0.02,
        "verdict": "pass" if err <= 0.02 else "fail",
    }


def t3_pixel_convergence(img8):
    dx_c, n_c = 0.4e-3, 256
    dx_f, n_f = 0.2e-3, 512
    ic = intensity(encode(img8, n_c, dx_c), dx_c)
    i_f = intensity(encode(img8, n_f, dx_f), dx_f)
    i_f_avg = i_f.reshape(n_c, 2, n_c, 2).mean(axis=(1, 3))   # 2x2 block average -> dx_c grid
    cc = ic[64:192, 64:192]
    cf = i_f_avg[64:192, 64:192]
    err = float(np.linalg.norm(cc - cf) / np.linalg.norm(cf))
    # Exploratory, added AFTER the pre-registered value was observed (see contract registry).
    # Fields are normalised to unit energy per grid, so a coarse pixel holds the SUM of its
    # four fine pixels. The pre-registered block mean differs from that by a factor of 4.
    i_f_sum = i_f.reshape(n_c, 2, n_c, 2).sum(axis=(1, 3))
    err_sum = float(np.linalg.norm(cc - i_f_sum[64:192, 64:192]) / np.linalg.norm(i_f_sum[64:192, 64:192]))
    return {
        "coarse": {"dx_m": dx_c, "N": n_c, "validity_limit_m": n_c * dx_c * dx_c / LAM},
        "fine": {"dx_m": dx_f, "N": n_f, "validity_limit_m": n_f * dx_f * dx_f / LAM},
        "window_m": n_c * dx_c,
        "rel_l2_intensity": err,
        "threshold": 0.05,
        "verdict": "pass" if err <= 0.05 else "fail",
        "exploratory_rel_l2_energy_sum": err_sum,
        "exploratory_note": "post hoc; not the pre-registered verdict",
    }


def main():
    digits = load_digits()
    img8 = digits.images[np.where(digits.target == 3)[0][0]] / 16.0
    result = {
        "contract": "CONTRATO-004",
        "lambda_m": LAM,
        "distance_m": D,
        "T1": t1_tilted_plane_waves(),
        "T2": t2_zero_padding(img8),
        "T3": t3_pixel_convergence(img8),
    }
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    (OUT_LOCAL / "regime_d2nn.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k != "T1"}
    summary["T1"] = {k: v for k, v in result["T1"].items() if k != "per_k_rel_error"}
    (OUT_REPO / "regime_d2nn_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for key in ("T1", "T2", "T3"):
        r = result[key]
        value = r.get("max_rel_error", r.get("rel_l2_intensity"))
        print(f"{key}: value={value:.3e} threshold={r['threshold']} verdict={r['verdict']}")


if __name__ == "__main__":
    main()
