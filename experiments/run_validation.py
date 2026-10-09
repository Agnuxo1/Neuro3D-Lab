"""CONTRATO-002: validación del propagador frente a soluciones analíticas.

Ejecuta E1 (haz gaussiano), E2 (autoimagen de Talbot), E3 (apertura circular en el eje)
y E4 (exploratorio: dominio de validez). Los umbrales son los del contrato y no se cambian aquí.

Uso: python experiments/run_validation.py
Salida: resumen en experiments/results/validation_summary.json (pequeño, versionado) y
datos completos en $NEURO3D_LAB_OUT/results/validation.json (no versionado).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlab.propagation import angular_spectrum  # noqa: E402

LAM = 633e-9
# Full data goes outside the repo. NEURO3D_LAB_OUT selects the base folder; the default
# matches the Blender add-on: <system temp>/neuro3d-lab.
_OUT_BASE = os.environ.get("NEURO3D_LAB_OUT") or str(Path(tempfile.gettempdir()) / "neuro3d-lab")
OUT_LOCAL = Path(_OUT_BASE) / "results"
OUT_REPO = ROOT / "experiments" / "results"


def zmax(n, dx, lam=LAM):
    """Validity limit of plain angular spectrum: z < N dx^2 / lambda."""
    return n * dx * dx / lam


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def e1_gaussian():
    w0, n, dx = 20e-6, 1023, 2e-6
    x = (np.arange(n) - (n - 1) / 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    u0 = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    E0 = np.sum(np.abs(u0) ** 2) * dx * dx
    zR = math.pi * w0**2 / LAM
    k = 2 * math.pi / LAM
    c = n // 2
    rows = []
    for f in (0.25, 0.5, 1.0, 2.0):
        z = f * zR
        u = angular_spectrum(u0, dx, LAM, z)
        I = np.abs(u) ** 2
        Ix = I.sum(axis=0)  # marginal along x
        var_x = np.sum(Ix * x**2) / np.sum(Ix)
        w_num = 2 * math.sqrt(var_x)  # intensity ~ exp(-2 r^2 / w^2) -> var = w^2/4
        w_an = w0 * math.sqrt(1 + (z / zR) ** 2)
        amp_num = abs(u[c, c])
        amp_an = 1 / math.sqrt(1 + (z / zR) ** 2)
        phase_num = wrap(np.angle(u[c, c]) - k * z)
        phase_an = wrap(-math.atan(z / zR))
        E = np.sum(I) * dx * dx
        rows.append({
            "z_over_zR": f, "z_m": z, "z_max_m": zmax(n, dx),
            "in_domain": bool(z < zmax(n, dx)),
            "w_rel_err": abs(w_num / w_an - 1),
            "amp_rel_err": abs(amp_num / amp_an - 1),
            "gouy_phase_err_rad": abs(wrap(phase_num - phase_an)),
            "energy_rel_err": abs(E / E0 - 1),
        })
    ok = all(r["in_domain"] and r["w_rel_err"] <= 0.005 and r["amp_rel_err"] <= 0.005
             and r["gouy_phase_err_rad"] <= 0.005 and r["energy_rel_err"] <= 1e-9 for r in rows)
    return {"id": "E1", "name": "haz gaussiano y fase de Gouy", "rows": rows,
            "verdict": "cumple" if ok else "no cumple"}


def e2_talbot():
    p, n, dx = 16e-6, 1024, 2e-6
    x = np.arange(n) * dx
    t = 0.5 * (1 + np.cos(2 * math.pi * x / p))
    u0 = np.tile(t, (n, 1)).astype(complex)  # periodic along x, uniform along y
    zT = 2 * p * p / LAM
    I0 = np.abs(u0) ** 2
    I_T = np.abs(angular_spectrum(u0, dx, LAM, zT)) ** 2
    I_half = np.abs(angular_spectrum(u0, dx, LAM, zT / 2)) ** 2
    e_T = np.linalg.norm(I_T - I0) / np.linalg.norm(I0)
    e_half = np.linalg.norm(I_half - I0) / np.linalg.norm(I0)
    ok = (zT < zmax(n, dx)) and e_T <= 0.01 and e_half >= 0.5
    return {"id": "E2", "name": "autoimagen de Talbot", "zT_m": zT, "z_max_m": zmax(n, dx),
            "rel_rms_at_zT": e_T, "rel_rms_at_zT_half": e_half,
            "verdict": "cumple" if ok else "no cumple"}


def e3_aperture():
    a_nom, n, dx = 0.2e-3, 1023, 8e-6
    x = (np.arange(n) - (n - 1) / 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    mask = (X**2 + Y**2 <= a_nom**2)
    u0 = mask.astype(complex)
    # radio efectivo del área muestreada (declarado en el contrato)
    a_eff = math.sqrt(mask.sum() * dx * dx / math.pi)
    c = n // 2
    rows = []
    for z in (0.02, 0.04, 0.08):
        I_on = abs(angular_spectrum(u0, dx, LAM, z)[c, c]) ** 2
        NF = a_eff**2 / (LAM * z)
        I_an = 4 * math.sin(math.pi * NF / 2) ** 2
        rows.append({"z_m": z, "z_max_m": zmax(n, dx), "in_domain": bool(z < zmax(n, dx)),
                     "NF": NF, "I_num": I_on, "I_an": I_an, "abs_err": abs(I_on - I_an)})
    ok = all(r["in_domain"] and r["abs_err"] <= 0.05 for r in rows)
    return {"id": "E3", "name": "apertura circular, intensidad en el eje", "a_eff_m": a_eff,
            "rows": rows, "verdict": "cumple" if ok else "no cumple"}


def e4_domain():
    """Exploratorio: error de amplitud y anchura en el eje frente a z/z_max (sin veredicto)."""
    w0, n, dx = 20e-6, 1023, 2e-6
    x = (np.arange(n) - (n - 1) / 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    u0 = np.exp(-(X**2 + Y**2) / w0**2).astype(complex)
    zR = math.pi * w0**2 / LAM
    zm = zmax(n, dx)
    rows = []
    for mult in (0.5, 1.0, 2.0):
        z = mult * zm
        u = angular_spectrum(u0, dx, LAM, z)
        I = np.abs(u) ** 2
        Ix = I.sum(axis=0)
        w_num = 2 * math.sqrt(np.sum(Ix * x**2) / np.sum(Ix))
        w_an = w0 * math.sqrt(1 + (z / zR) ** 2)
        amp_an = 1 / math.sqrt(1 + (z / zR) ** 2)
        rows.append({"z_over_zmax": mult, "z_m": z,
                     "w_rel_err": abs(w_num / w_an - 1),
                     "amp_rel_err": abs(abs(u[n // 2, n // 2]) / amp_an - 1)})
    return {"id": "E4", "name": "dominio de validez (exploratorio, sin veredicto)", "rows": rows}


def main():
    results = {"contract": "CONTRATO-002", "lambda_m": LAM,
               "experiments": [e1_gaussian(), e2_talbot(), e3_aperture(), e4_domain()]}
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    (OUT_LOCAL / "validation.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summary = {"contract": "CONTRATO-002",
               "verdicts": {e["id"]: e.get("verdict", "exploratorio") for e in results["experiments"]}}
    (OUT_REPO / "validation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for e in results["experiments"]:
        print(e["id"], e["name"], "->", e.get("verdict", "exploratorio"))
        for r in e.get("rows", []):
            print("   ", {k: (float(f"{v:.3g}") if isinstance(v, float) else v) for k, v in r.items()})
        for k in ("rel_rms_at_zT", "rel_rms_at_zT_half"):
            if k in e:
                print("   ", k, float(f"{e[k]:.3g}"))


if __name__ == "__main__":
    main()
