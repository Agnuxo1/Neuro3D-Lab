"""Negative control for CONTRATO-002: a reversed propagation direction must fail the Gouy-phase check.

Propagates a Gaussian beam by +z and by -z, compares the on-axis Gouy phase with the analytic
value -arctan(z/z_R), and writes the errors to experiments/results/negative_control.json.
Run: python experiments/negative_control.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from optlab.propagation import angular_spectrum  # noqa: E402

LAM, W0, N, DX = 633e-9, 20e-6, 1023, 2e-6


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def main():
    x = (np.arange(N) - (N - 1) / 2) * DX
    X, Y = np.meshgrid(x, x)
    u0 = np.exp(-(X**2 + Y**2) / W0**2).astype(complex)
    zR = math.pi * W0**2 / LAM
    k = 2 * math.pi / LAM
    c = N // 2
    rows = []
    for f in (0.25, 1.0):
        z = f * zR
        for sign in (+1, -1):
            u = angular_spectrum(u0, DX, LAM, sign * z)
            phase = wrap(np.angle(u[c, c]) - sign * k * z)
            expected = wrap(-math.atan(z / zR))
            rows.append({"z_over_zR": f, "direction": sign, "gouy_error_rad": abs(wrap(phase - expected)),
                         "threshold_rad": 0.005, "fails_check": bool(abs(wrap(phase - expected)) > 0.005)})
    out = ROOT / "experiments" / "results" / "negative_control.json"
    out.write_text(json.dumps({"contract": "CONTRATO-002", "control": "reversed propagation", "rows": rows}, indent=1),
                   encoding="utf-8")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
