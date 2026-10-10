"""POST HOC diagnostic for CONTRATO-006 V2 (not a verdict; see docs/CONTRACTS/CONTRATO-006b-diagnostic.md).

Full 128 x 128 grid comparison of the tapered tilted wave I2 at d = 18 mm:
  ASM (N = 128)  against  RS with the CONTRATO-006 window (51.2 mm, source pitch dx/2)
  ASM (N = 128)  against  RS with a 205 mm window (source pitch dx/2)
  RS 51.2 mm     against  RS 205 mm
Also reports the relative error in radial bands, to show where the discrepancy sits.
Usage: python -B experiments/diagnose_v2_fullgrid.py
Output: experiments/results/diagnose_v2_fullgrid.json
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
SIN = 0.498047            # DFT-bin tilt used by CONTRATO-006 (I2)
DX = 0.4e-3
N = 128


def taper_tilt(x, y):
    rho = np.sqrt(x**2 + y**2)
    return np.exp(-((rho / 18e-3) ** 8)) * np.exp(1j * K * SIN * x)


def rs_on_window(n_src, obs_x, obs_y):
    ps = DX / 2
    c = (np.arange(n_src) - (n_src - 1) / 2) * ps
    XS, YS = np.meshgrid(c, c)
    fields = taper_tilt(XS, YS).ravel()[:, None]
    out = R.rs_apply(obs_x.ravel(), obs_y.ravel(), XS.ravel(), YS.ravel(), ps * ps, D, fields,
                     kinds=("exact",))["exact"][:, 0]
    return out


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    t0 = time.time()
    c = (np.arange(N) - (N - 1) / 2) * DX
    GX, GY = np.meshgrid(c, c)
    asm = angular_spectrum(taper_tilt(GX, GY), DX, LAM, D).ravel()
    ox, oy = GX.ravel(), GY.ravel()
    rs51 = rs_on_window(256, ox, oy)              # the CONTRATO-006 reference window
    print(f"RS 51.2 mm done in {time.time() - t0:.1f} s", flush=True)
    rs205 = rs_on_window(1024, ox, oy)            # 205 mm window
    print(f"RS 205 mm done in {time.time() - t0:.1f} s", flush=True)
    r = np.sqrt(ox**2 + oy**2)
    bands = {"r<=6.4mm": r <= 6.4e-3, "6.4<r<=12.8mm": (r > 6.4e-3) & (r <= 12.8e-3),
             "12.8<r<=25.6mm": (r > 12.8e-3) & (r <= 25.6e-3)}
    res = {
        "contract": "CONTRATO-006 V2 diagnostic (post hoc)",
        "full_grid_nodes": int(ox.size),
        "rel_asm_vs_rs51": rel(asm, rs51),
        "rel_asm_vs_rs205": rel(asm, rs205),
        "rel_rs51_vs_rs205": rel(rs51, rs205),
        "bands_asm_vs_rs205": {k: rel(asm[m], rs205[m]) for k, m in bands.items()},
        "bands_asm_vs_rs51": {k: rel(asm[m], rs51[m]) for k, m in bands.items()},
        "runtime_s": round(time.time() - t0, 1),
    }
    out = ROOT / "experiments" / "results" / "diagnose_v2_fullgrid.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
