"""CONTRACT-006 (docs/CONTRACTS/CONTRATO-006-rayleigh-sommerfeld.md).

Direct-summation Rayleigh-Sommerfeld (RS) reference for the scalar propagator in the
D2NN sampling regime (dx/lambda = 0.53). The reference is independent of the angular
spectrum (ASM) used by optlab.propagation.

Usage:
  python experiments/run_rayleigh_sommerfeld.py --sanity   # gate S0 only (32x32, fast)
  python experiments/run_rayleigh_sommerfeld.py            # full pre-registered run

The full run first checks the S0 gate and stops if it fails.
Outputs of the full run:
  experiments/results/rayleigh_sommerfeld_summary.json  (values and verdicts only)
  $NEURO3D_LAB_OUT/results/rayleigh_sommerfeld.json     (full record; default
      <repo parent>/.cognition/neuro3d-lab/results)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import exp1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlab.propagation import angular_spectrum  # noqa: E402

# ---- fixed parameters (CONTRATO-006) ----
LAM = 0.75e-3                    # vacuum wavelength [m]; medium index 1
K = 2.0 * np.pi / LAM            # wavenumber [1/m]
DX = 0.4e-3                      # coarse pitch [m]
N = 128                          # coarse grid; window N*DX = 51.2 mm
D = 18e-3                        # propagation distance [m]
W_GAUSS = 3.0e-3                 # I1 Gaussian width [m]
DIGIT_CLASS = 3                  # I3: first sample of class 3 in sklearn load_digits
RHO_TAPER = 18e-3                # I2 taper radius [m] (addendum A3)
TAPER_ORDER = 8                  # I2 super-Gaussian order (addendum A3)
M_BIN = 34                       # I2 DFT bin (addendum A3)
SIN_T = M_BIN * LAM / (N * DX)   # 0.498046875; nominal 0.5 is not a bin on this grid
COS_T = np.sqrt(1.0 - SIN_T**2)
N_SANITY = 32                    # S0 grid
MAX_KERNEL_ENTRIES = 4_000_000   # complex kernel entries per chunk (~64 MB per array)

THRESH = {"S0": 1e-3, "C1": 5e-3, "V1": 1e-2, "V2": 1e-2, "V3": 5e-2}

OUT_REPO = ROOT / "experiments" / "results"
OUT_LOCAL = Path(os.environ.get("NEURO3D_LAB_OUT")
                 or ROOT.parent / ".cognition" / "neuro3d-lab") / "results"


# ---- grids and kernels ----
def coords(n, pitch):
    return (np.arange(n) - (n - 1) / 2.0) * pitch


def grid(n, pitch):
    """Row i -> y_i, column j -> x_j, both centred (same convention as optlab.elements)."""
    c = coords(n, pitch)
    X, Y = np.meshgrid(c, c, indexing="xy")
    return X, Y


def rs_apply(xo, yo, xs, ys, dA, d, fields, kinds=("exact",)):
    """Direct RS sum over sources for every observation point, chunked over observation points.

    out[kind][p, m] = dA * sum_s K_kind(r_ps) * fields[s, m],  r_ps = sqrt(|P - Q_s|^2 + d^2).
    K_exact = (1/(i lam)) (d/r) (1 + i/(k r)) e^{ikr}/r   (= -(1/2pi) d_z(e^{ikr}/r))
    K_spec  = (1/(i lam)) (d/r) e^{ikr}/r                  (leading order, diagnostic only)
    """
    xo, yo, xs, ys = (np.ravel(a) for a in (xo, yo, xs, ys))
    fields = np.asarray(fields, dtype=complex)
    P, S = xo.size, xs.size
    out = {kind: np.empty((P, fields.shape[1]), dtype=complex) for kind in kinds}
    chunk = max(1, MAX_KERNEL_ENTRIES // S)
    for a in range(0, P, chunk):
        b = min(P, a + chunk)
        r = np.sqrt((xo[a:b, None] - xs[None, :]) ** 2 + (yo[a:b, None] - ys[None, :]) ** 2 + d * d)
        base = np.exp(1j * K * r) * (d / r) / (1j * LAM) / r
        for kind in kinds:
            kern = base * (1.0 + 1j / (K * r)) if kind == "exact" else base
            out[kind][a:b] = dA * (kern @ fields)
    return out


def closed_disc(R_eff):
    """Exact RS value on axis for a uniform unit input on a disc of radius R_eff (kernel K_exact).

    U = -(1/2pi) * 2pi * d_z int_0^R rho e^{ikr}/r drho = -d_z[(e^{ikr_R} - e^{ikz})/(ik)]
      = e^{ikz} - (z/r_R) e^{ikr_R},   r_R = sqrt(R^2 + z^2).
    """
    rR = np.sqrt(R_eff**2 + D * D)
    return np.exp(1j * K * D) - (D / rR) * np.exp(1j * K * rR)


# ---- gate S0 (and its diagnostics) ----
def s0_case(factor=2, kind="exact", geometry="disc", n=N_SANITY):
    """Uniform unit input on an n x n coarse window, source pitch DX/factor, observation on axis.

    geometry "disc":   source restricted to rho <= n*DX/2 (sampled area gives R_eff). Reference: closed_disc(R_eff).
    geometry "square": source on the full square window. Reference: exp(ikd) (diagnostic only).
    """
    dxs = DX / factor
    X, Y = grid(n * factor, dxs)
    R = n * DX / 2.0
    if geometry == "disc":
        m = (X**2 + Y**2) <= R**2
        R_eff = np.sqrt(int(m.sum()) * dxs**2 / np.pi)
        ref = closed_disc(R_eff)
    else:
        m = np.ones(X.shape, dtype=bool)
        R_eff = None
        ref = np.exp(1j * K * D)
    r = np.sqrt(X[m] ** 2 + Y[m] ** 2 + D * D)
    base = np.exp(1j * K * r) * (D / r) / (1j * LAM) / r
    kern = base * (1.0 + 1j / (K * r)) if kind == "exact" else base
    value = dxs * dxs * np.sum(kern)
    return {"value": complex(value), "ref": complex(ref), "error": float(abs(value - ref)),
            "R_eff_m": None if R_eff is None else float(R_eff), "n_sources": int(m.sum())}


def infinite_plane_checks():
    """Closed forms for a uniform input on the infinite plane (on axis).

    J = int_z^inf e^{ikr}/r dr = E1(-ikz). Spec kernel: -ikz J.
    Exact kernel adds (i/k) int_z^inf e^{ikr}/r^2 dr = (i/k)(e^{ikz}/z + ikJ), which gives e^{ikz} exactly.
    """
    x = K * D
    J = exp1(-1j * x)
    spec = -1j * x * J
    exact = spec + (np.exp(1j * x) + 1j * x * J)
    target = np.exp(1j * x)
    return {"spec_error": float(abs(spec - target)), "exact_error": float(abs(exact - target)),
            "spec_value_re_im": [float(spec.real), float(spec.imag)],
            "exact_value_re_im": [float(exact.real), float(exact.imag)]}


def run_sanity():
    gate = s0_case(factor=2, kind="exact", geometry="disc")
    d4 = s0_case(factor=4, kind="exact", geometry="disc")
    d8 = s0_case(factor=8, kind="exact", geometry="disc")
    spec_disc = s0_case(factor=2, kind="spec", geometry="disc")
    square = s0_case(factor=2, kind="exact", geometry="square")
    inf = infinite_plane_checks()
    passed = gate["error"] <= THRESH["S0"]

    print("CONTRACT-006 sanity: gate S0 only (no full run)")
    print(f"  grid N={N_SANITY}, window {N_SANITY * DX * 1e3:.1f} mm, lambda={LAM * 1e3:.2f} mm, d={D * 1e3:.1f} mm")
    print(f"  S0 kernel=exact, disc window R_eff={gate['R_eff_m'] * 1e3:.4f} mm, source pitch dx/2, "
          f"sources={gate['n_sources']}")
    print(f"  S0 normalisation error |U_RS - U_disc| = {gate['error']:.3e} (threshold {THRESH['S0']:.1e})")
    print(f"  S0 verdict = {'pass' if passed else 'FAIL'}")
    print("  diagnostics (not gates):")
    print(f"    D1 infinite plane, spec kernel error vs e^(ikd)   = {inf['spec_error']:.3e}")
    print(f"    D2 infinite plane, exact kernel closed-form error = {inf['exact_error']:.3e}")
    print(f"    D3 disc S0 at source pitch dx/4 = {d4['error']:.3e}; dx/8 = {d8['error']:.3e}")
    print(f"    D4 disc S0 with spec kernel      = {spec_disc['error']:.3e}")
    print(f"    D5 square window (no truncation correction) vs e^(ikd) = {square['error']:.3e}")
    print(f"  exit code = {0 if passed else 1}")
    return 0 if passed else 1


# ---- full run ----
def digit_fields():
    from sklearn.datasets import load_digits

    d = load_digits()
    img8 = d.images[np.where(d.target == DIGIT_CLASS)[0][0]] / 16.0
    up8 = np.kron(img8, np.ones((8, 8)))
    up16 = np.kron(img8, np.ones((16, 16)))
    norm = np.linalg.norm(up8)                       # unit energy on the coarse grid
    coarse = np.pad(up8 / norm, (N - 64) // 2).astype(complex)
    fine = np.pad(up16 / norm, (2 * N - 128) // 2).astype(complex)
    assert coarse.shape == (N, N) and fine.shape == (2 * N, 2 * N)
    return coarse, fine


def tilt_fields(X, Y):
    taper = np.exp(-((np.sqrt(X**2 + Y**2) / RHO_TAPER) ** TAPER_ORDER))
    wave = np.exp(1j * K * SIN_T * X)
    return taper * wave, wave                        # tapered (V2), untapered (diagnostic)


def gauss_field(X, Y):
    return np.exp(-(X**2 + Y**2) / W_GAUSS**2).astype(complex)


def rel_l2(a, b):
    """Relative L2 difference with b as the reference."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def intensity(u):
    return np.abs(u) ** 2


def run_full():
    t_start = time.time()
    if D >= N * DX * DX / LAM:
        raise SystemExit("ASM outside its validity domain for N, DX, D")
    gate = s0_case(factor=2, kind="exact", geometry="disc")
    if gate["error"] > THRESH["S0"]:
        print(f"S0 gate FAILED ({gate['error']:.3e} > {THRESH['S0']:.1e}); full run not started")
        return 1
    print(f"S0 gate passed ({gate['error']:.3e}); starting full run")

    Xc, Yc = grid(N, DX)
    Xf2, Yf2 = grid(2 * N, DX / 2)
    Xf4, Yf4 = grid(4 * N, DX / 4)

    # Coarse inputs (for ASM)
    g_c = gauss_field(Xc, Yc)
    t_c, u_c = tilt_fields(Xc, Yc)
    dig_c, dig_f = digit_fields()

    # Fine inputs for RS at dx/2
    g_f2 = gauss_field(Xf2, Yf2)
    t_f2, u_f2 = tilt_fields(Xf2, Yf2)
    F2 = np.stack([g_f2.ravel(), t_f2.ravel(), dig_f.ravel(), u_f2.ravel()], axis=1)

    # RS at dx/2 (exact and spec kernels), observation on the coarse grid
    t0 = time.time()
    out2 = rs_apply(Xc, Yc, Xf2, Yf2, (DX / 2) ** 2, D, F2, kinds=("exact", "spec"))
    t_rs2 = time.time() - t0
    rs2 = {k: v.reshape(N, N, 4) for k, v in out2.items()}

    # RS at dx/4 for C1 (I1 only)
    g_f4 = gauss_field(Xf4, Yf4)
    t0 = time.time()
    out4 = rs_apply(Xc, Yc, Xf4, Yf4, (DX / 4) ** 2, D, g_f4.ravel()[:, None], kinds=("exact",))
    t_rs4 = time.time() - t0
    rs4 = out4["exact"].reshape(N, N)

    # ASM on the coarse grid
    stack = np.stack([g_c, t_c, dig_c, u_c])
    asm = angular_spectrum(stack, DX, LAM, D)        # shape (4, N, N)

    exact_tilt = np.exp(1j * (K * SIN_T * Xc + K * COS_T * D))
    values = {
        "V1_I1_intensity_ASM_vs_RS": rel_l2(intensity(asm[0]), intensity(rs2["exact"][..., 0])),
        "C1_I1_intensity_RS_dx2_vs_dx4": rel_l2(intensity(rs2["exact"][..., 0]), intensity(rs4)),
        "V2_I2_tapered_field_ASM_vs_RS": rel_l2(asm[1], rs2["exact"][..., 1]),
        "V3_I3_intensity_ASM_vs_RS": rel_l2(intensity(asm[2]), intensity(rs2["exact"][..., 2])),
    }
    diagnostics = {
        "D1_I2_untapered_ASM_vs_exact_tilt": rel_l2(asm[3], exact_tilt),
        "D2_I2_untapered_RS_vs_exact_tilt": rel_l2(rs2["exact"][..., 3], exact_tilt),
        "D3_I2_untapered_ASM_vs_RS": rel_l2(asm[3], rs2["exact"][..., 3]),
        "D4_I1_intensity_spec_kernel_vs_exact": rel_l2(intensity(rs2["spec"][..., 0]), intensity(rs2["exact"][..., 0])),
        "D5_I1_intensity_ASM_vs_RS_spec_kernel": rel_l2(intensity(asm[0]), intensity(rs2["spec"][..., 0])),
        "D6_S0_spec_kernel_disc": s0_case(factor=2, kind="spec", geometry="disc")["error"],
        "D7_S0_square_window_vs_exp_ikd": s0_case(factor=2, kind="exact", geometry="square")["error"],
        "D8_infinite_plane": infinite_plane_checks(),
    }
    verdicts = {
        "S0": "pass",
        "C1": "pass" if values["C1_I1_intensity_RS_dx2_vs_dx4"] <= THRESH["C1"] else "fail",
        "V1": "pass" if values["V1_I1_intensity_ASM_vs_RS"] <= THRESH["V1"] else "fail",
        "V2": "pass" if values["V2_I2_tapered_field_ASM_vs_RS"] <= THRESH["V2"] else "fail",
        "V3": "pass" if values["V3_I3_intensity_ASM_vs_RS"] <= THRESH["V3"] else "fail",
    }
    elapsed = time.time() - t_start

    summary = {
        "contract": "CONTRATO-006",
        "status": "full run",
        "amendments": ["A1 exact RS kernel", "A2 S0 on disc window with closed form",
                       "A3 I2 bin-aligned tilt with taper for V2"],
        "thresholds": THRESH,
        "gate_S0_error": gate["error"],
        "values": values,
        "verdicts": verdicts,
    }
    record = {
        **summary,
        "parameters": {"lambda_m": LAM, "dx_m": DX, "N": N, "d_m": D, "w_gauss_m": W_GAUSS,
                       "digit_class": DIGIT_CLASS, "rho_taper_m": RHO_TAPER, "taper_order": TAPER_ORDER,
                       "sin_theta": SIN_T, "zmax_asm_m": N * DX * DX / LAM},
        "diagnostics": diagnostics,
        "timing_s": {"rs_dx2": t_rs2, "rs_dx4": t_rs4, "total": elapsed},
        "chunk_budget_entries": MAX_KERNEL_ENTRIES,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
    }

    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_LOCAL / "rayleigh_sommerfeld_fields.npz",
                        asm=asm, rs_dx2=rs2["exact"], rs_dx4_gauss=rs4)
    (OUT_LOCAL / "rayleigh_sommerfeld.json").write_text(json.dumps(record, indent=1, default=float),
                                                        encoding="utf-8")
    (OUT_REPO / "rayleigh_sommerfeld_summary.json").write_text(
        json.dumps(summary, indent=1, default=float), encoding="utf-8")
    for key, val in values.items():
        print(f"{key}: {val:.4e}")
    for key, verdict in verdicts.items():
        print(f"{key} verdict: {verdict}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="CONTRACT-006 Rayleigh-Sommerfeld reference")
    parser.add_argument("--sanity", action="store_true", help="run gate S0 only and exit")
    args = parser.parse_args()
    sys.exit(run_sanity() if args.sanity else run_full())


if __name__ == "__main__":
    main()
