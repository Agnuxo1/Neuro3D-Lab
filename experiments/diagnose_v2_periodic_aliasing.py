"""POST HOC diagnostic for CONTRATO-006 V2 (not a verdict; see docs/CONTRACTS/CONTRATO-006c-periodic-sampling.md).

Separates two candidate mechanisms for the 0.16 annulus discrepancy of CONTRATO-006b addendum 2:
  (A) periodic images of the 51.2 mm FFT window  -> R3: same dx, zero-padded window of 102.4 mm
  (B) input sampling at dx                        -> R4: same 51.2 mm period, input sampled at dx/2
Reference R2: RS with a 51.2 mm source window at source pitch dx/2 (the CONTRATO-006 V2 reference).
Usage: python -B experiments/diagnose_v2_periodic_aliasing.py
Output: experiments/results/diagnose_v2_periodic_aliasing.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import run_rayleigh_sommerfeld as R  # noqa: E402
from optlab.propagation import angular_spectrum  # noqa: E402

LAM = 0.75e-3
K = 2 * math.pi / LAM
D = 18e-3
SIN = 0.498047            # DFT-bin tilt used by CONTRATO-006 (I2): 34 bins over 51.2 mm
DX = 0.4e-3
N = 128
V2_REGISTERED = 0.1334889131246678   # CONTRATO-006b addendum 2, rel(ASM, RS 51.2 mm)
TOL_V2 = 1e-2
TOL_REFUTE = 5e-2


def taper_tilt(x, y):
    rho = np.sqrt(x**2 + y**2)
    return np.exp(-((rho / 18e-3) ** 8)) * np.exp(1j * K * SIN * x)


def rs_on_window(n_src, obs_x, obs_y):
    ps = DX / 2
    c = (np.arange(n_src) - (n_src - 1) / 2) * ps
    XS, YS = np.meshgrid(c, c)
    fields = taper_tilt(XS, YS).ravel()[:, None]
    return R.rs_apply(obs_x, obs_y, XS.ravel(), YS.ravel(), ps * ps, D, fields,
                      kinds=("exact",))["exact"][:, 0]


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    t0 = time.time()
    c = (np.arange(N) - (N - 1) / 2) * DX
    GX, GY = np.meshgrid(c, c)
    ox, oy = GX.ravel(), GY.ravel()
    r = np.sqrt(ox**2 + oy**2)
    bands = {"r<=6.4mm": r <= 6.4e-3, "6.4<r<=12.8mm": (r > 6.4e-3) & (r <= 12.8e-3),
             "12.8<r<=25.6mm": (r > 12.8e-3) & (r <= 25.6e-3)}
    annulus = bands["12.8<r<=25.6mm"]

    # R1: ASM on the registered 128 grid (periodic 51.2 mm window)
    r1 = angular_spectrum(taper_tilt(GX, GY), DX, LAM, D).ravel()
    # R2: RS reference, 51.2 mm source window, source pitch dx/2 (CONTRATO-006 V2 reference)
    r2 = rs_on_window(256, ox, oy)
    print(f"R1 and R2 done in {time.time() - t0:.1f} s", flush=True)

    # R3: ASM at dx on a 256 grid (window 102.4 mm): periodic images removed. Central 128 nodes.
    c256 = (np.arange(256) - 127.5) * DX
    X3, Y3 = np.meshgrid(c256, c256)
    r3_full = angular_spectrum(taper_tilt(X3, Y3), DX, LAM, D)
    r3 = r3_full[64:192, 64:192].ravel()

    # R4: ASM at dx/2 on a 256 grid with the same 51.2 mm period as R1. Odd nodes m = 2n + 1 = the R1 nodes.
    cf = (np.arange(256) - 128) * DX / 2
    XF, YF = np.meshgrid(cf, cf)
    r4_full = angular_spectrum(taper_tilt(XF, YF), DX / 2, LAM, D)
    r4 = r4_full[1::2, 1::2].ravel()
    # position check: odd fine node m = 2n + 1 must sit at the R1 node (n - 63.5) * dx
    assert np.allclose(cf[1::2], c), "fine odd nodes do not coincide with the 128 nodes"
    print(f"R3 and R4 done in {time.time() - t0:.1f} s", flush=True)

    q1 = rel(r1, r2)
    q1_ok = abs(q1 - V2_REGISTERED) <= 1e-6
    q2 = rel(r3, r2)
    q3 = rel(r4, r2)
    q2_ann = rel(r3[annulus], r2[annulus])
    q3_ann = rel(r4[annulus], r2[annulus])
    images_explain = q2 <= TOL_V2 and q2_ann <= TOL_V2
    sampling_explains = q3 <= TOL_V2 and q3_ann <= TOL_V2
    if not q1_ok:
        outcome = "invalid: Q1 did not reproduce the registered V2 value"
    elif images_explain and sampling_explains:
        outcome = "both (inconsistency to resolve)"
    elif images_explain:
        outcome = "periodic images explain the discrepancy"
    elif sampling_explains:
        outcome = "input sampling explains the discrepancy"
    elif q2_ann >= TOL_REFUTE and q3_ann >= TOL_REFUTE:
        outcome = "neither: scalar kernel or ASM model remains (open hypothesis)"
    else:
        outcome = "inconclusive"

    edge = np.abs(taper_tilt(GX, GY))
    edge_max = max(edge[0].max(), edge[-1].max(), edge[:, 0].max(), edge[:, -1].max())
    res = {
        "contract": "CONTRATO-006c (post hoc)",
        "q1_asm128_vs_rs51_full": q1,
        "q1_reproduced_registered_v2": bool(q1_ok),
        "q2_images_asm256pad_vs_rs51_full": q2,
        "q2_images_asm256pad_vs_rs51_annulus": q2_ann,
        "q3_sampling_asm_dx2_vs_rs51_full": q3,
        "q3_sampling_asm_dx2_vs_rs51_annulus": q3_ann,
        "q4_decomp_asm128_vs_asm256pad_full": rel(r1, r3),
        "q5_decomp_asm128_vs_asm_dx2_full": rel(r1, r4),
        "bands_q2_images": {k: rel(r3[m], r2[m]) for k, m in bands.items()},
        "bands_q3_sampling": {k: rel(r4[m], r2[m]) for k, m in bands.items()},
        "bands_q1_asm128": {k: rel(r1[m], r2[m]) for k, m in bands.items()},
        "decision": {"images_explain": bool(images_explain), "sampling_explains": bool(sampling_explains),
                     "tol_v2": TOL_V2, "tol_refute": TOL_REFUTE},
        "outcome": outcome,
        "descriptive_max_abs_input_on_128_edge": float(edge_max),
        "runtime_s": round(time.time() - t0, 1),
    }
    out = ROOT / "experiments" / "results" / "diagnose_v2_periodic_aliasing.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
