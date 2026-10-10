"""POST HOC, DEV only: sensitivity of the seed-0 D2NN DEV predictions to the periodic FFT window (C12).

Pre-registered in docs/CONTRACTS/CONTRATO-010-c12-periodic-dev.md (written before any run).
Compares the registered periodic propagation (P, 128 grid, 51.2 mm window) with a zero-padded
propagation (Z, 256 grid, 102.4 mm window) on the 359 DEV images only. TEST is not encoded,
not loaded for evaluation and not evaluated. optlab is not modified.

Usage: python -B experiments/c12_periodic_dev_d2nn.py
Input: experiments/results/d2nn_seed0_phases.npz (written by experiments/save_d2nn_seed0.py;
       this script does not train).
Output: experiments/results/c12_periodic_dev_d2nn.json, written only if Q1 reproduces.
        If Q1 fails, the script prints the failure, exits with status 2 and writes no verdict.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # no __pycache__ inside the repository

EXPERIMENTS = Path(__file__).resolve().parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT / "src"))

import run_d2nn_digits as R  # noqa: E402  (importing does not run main())
from optlab.propagation import angular_spectrum  # noqa: E402

PHASES_PATH = EXPERIMENTS / "results" / "d2nn_seed0_phases.npz"
CHECKSUM_PATH = EXPERIMENTS / "results" / "d2nn_seed0_checksum.json"
OUT_PATH = EXPERIMENTS / "results" / "c12_periodic_dev_d2nn.json"

BETA = 100.0
SEED = 0
N = R.N                       # 128: registered grid, periodic window 51.2 mm
NZ = 256                      # zero-padded grid, window 102.4 mm
OFF = (NZ - N) // 2           # 64: offset of the 128 window inside the 256 grid
L = len(R.DIST) - 1           # 3 phase layers
N_DEV_EXPECTED = 359

# ---- thresholds, fixed in CONTRATO-010 before any run ----
EXPECTED_DEV_ACC = 0.9554317548746518
Q1_TOL = 1e-9
Q2_NEG, Q2_MAT = 0.005, 0.02   # abs change in DEV accuracy
Q3_NEG, Q3_MAT = 0.01, 0.03    # fraction of DEV predictions whose class changes
Q4_NEG, Q4_MAT = 0.01, 0.05    # median over images of relative L2 change of the 10 scores


def load_phases():
    if not PHASES_PATH.exists():
        raise SystemExit(f"Missing {PHASES_PATH}. Run experiments/save_d2nn_seed0.py first. "
                         "This script does not train.")
    data = PHASES_PATH.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    with np.load(PHASES_PATH) as z:
        phases = np.array(z["phases"], dtype=float)
        beta = float(z["beta"])
        seed = int(z["seed"])
        pixel_m = float(z["pixel_m"])
        wavelength_m = float(z["wavelength_m"])
        distances_m = np.array(z["distances_m"], dtype=float)
    checks = {
        "beta": beta == BETA,
        "seed": seed == SEED,
        "pixel_m": pixel_m == R.DX,
        "wavelength_m": wavelength_m == R.LAM,
        "distances_m": np.array_equal(distances_m, np.asarray(R.DIST, dtype=float)),
        "shape": phases.shape == (L, N, N),
    }
    if not all(checks.values()):
        raise SystemExit(f"Metadata of {PHASES_PATH.name} does not match the contract: {checks}")
    matches_checksum_file = None
    if CHECKSUM_PATH.exists():
        recorded = json.loads(CHECKSUM_PATH.read_text(encoding="utf-8"))
        matches_checksum_file = recorded.get("sha256") == sha
    return phases, sha, matches_checksum_file


def padded_phase_exp(phases):
    """Phase 0 (transparent) outside the central 128 window of the 256 grid."""
    ph = np.zeros((L, NZ, NZ), dtype=float)
    ph[:, OFF:OFF + N, OFF:OFF + N] = phases
    return np.exp(1j * ph)


def forward_zero_padded(u128, ph_exp256, masks128):
    a = np.zeros((NZ, NZ), dtype=complex)
    a[OFF:OFF + N, OFF:OFF + N] = u128                      # input centred in the 256 grid
    for k in range(L):
        a = angular_spectrum(a, R.DX, R.LAM, R.DIST[k]) * ph_exp256[k]
    f = angular_spectrum(a, R.DX, R.LAM, R.DIST[L])
    I_central = np.abs(f[OFF:OFF + N, OFF:OFF + N]) ** 2     # read from the central 128 window
    return BETA * np.tensordot(masks128, I_central, axes=([1, 2], [0, 1]))


def classify(x, neg_max, mat_min):
    if x <= neg_max:
        return "negligible"
    if x >= mat_min:
        return "material"
    return "intermediate"


def main():
    t0 = time.time()
    img8, y_all, tr, dv, _te_unused = R.load_split()
    del _te_unused                                          # TEST indices are discarded, never encoded
    if len(dv) != N_DEV_EXPECTED:
        raise SystemExit(f"DEV size {len(dv)} != {N_DEV_EXPECTED}")
    y_dv = y_all[dv]
    U_dv = [R.encode_field(img8[i]) for i in dv]           # encoded as in run_d2nn_digits.py

    phases, npz_sha, matches_checksum_file = load_phases()
    masks128 = R.build_masks()
    model = R.make_model(masks128, BETA, seed=SEED)        # registered DiffractiveNetwork (optlab)
    model.phases = phases.copy()

    # ---- P: registered periodic propagation ----
    s_P = np.array([model.forward(u)["scores"] for u in U_dv])   # (359, 10)
    pred_P = np.argmax(s_P, axis=1)                              # same rule as model.predict
    acc_P = R.acc(pred_P, y_dv)

    # ---- Q1 gate: no verdict if the registered DEV accuracy is not reproduced ----
    q1_diff = abs(acc_P - EXPECTED_DEV_ACC)
    if not (q1_diff <= Q1_TOL):
        print(f"Q1 FAILED: periodic DEV acc {acc_P!r} vs {EXPECTED_DEV_ACC!r} "
              f"(|diff|={q1_diff!r} > {Q1_TOL}). Run invalid; no verdict written.",
              file=sys.stderr, flush=True)
        sys.exit(2)

    # ---- Z: zero-padded propagation ----
    ph_exp256 = padded_phase_exp(phases)
    s_Z = np.array([forward_zero_padded(u, ph_exp256, masks128) for u in U_dv])
    pred_Z = np.argmax(s_Z, axis=1)
    acc_Z = R.acc(pred_Z, y_dv)

    # ---- metrics ----
    d_acc = float(abs(acc_P - acc_Z))                                   # Q2
    changed = pred_P != pred_Z
    frac_changed = float(np.mean(changed))                              # Q3
    norm_P = np.linalg.norm(s_P, axis=1)
    if np.any(norm_P == 0.0):
        raise SystemExit("Zero periodic score vector; Q4 is undefined.")
    rel = np.linalg.norm(s_Z - s_P, axis=1) / norm_P
    med_rel = float(np.median(rel))                                     # Q4

    c_cls = {
        "Q2": classify(d_acc, Q2_NEG, Q2_MAT),
        "Q3": classify(frac_changed, Q3_NEG, Q3_MAT),
        "Q4": classify(med_rel, Q4_NEG, Q4_MAT),
    }
    if "material" in c_cls.values():
        outcome = "material"
    elif all(v == "negligible" for v in c_cls.values()):
        outcome = "negligible"
    else:
        outcome = "inconclusive"

    correct_P = pred_P == y_dv
    correct_Z = pred_Z == y_dv
    result = {
        "contract": "CONTRATO-010",
        "status": "POST HOC, DEV only, TEST not used",
        "input": {
            "phases_file": PHASES_PATH.name,
            "sha256": npz_sha,
            "matches_checksum_file": matches_checksum_file,
            "beta": BETA,
            "seed": SEED,
            "n_dev": int(len(dv)),
        },
        "q1": {"expected": EXPECTED_DEV_ACC, "periodic_dev_acc": float(acc_P),
               "abs_diff": float(q1_diff), "tolerance": Q1_TOL, "pass": True},
        "zero_padded": {"grid": NZ, "offset": OFF, "dev_acc": float(acc_Z)},
        "q2": {"abs_diff_dev_acc": d_acc, "negligible_max": Q2_NEG,
               "material_min": Q2_MAT, "class": c_cls["Q2"]},
        "q3": {"fraction_changed": frac_changed, "n_changed": int(np.sum(changed)),
               "negligible_max": Q3_NEG, "material_min": Q3_MAT, "class": c_cls["Q3"]},
        "q4": {"median_rel_L2": med_rel, "p90_rel_L2": float(np.percentile(rel, 90)),
               "max_rel_L2": float(np.max(rel)), "negligible_max": Q4_NEG,
               "material_min": Q4_MAT, "class": c_cls["Q4"]},
        "descriptive": {"correct_periodic_only": int(np.sum(correct_P & ~correct_Z)),
                        "correct_zero_padded_only": int(np.sum(~correct_P & correct_Z))},
        "outcome": outcome,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"Q1 ok: periodic DEV acc={acc_P:.16f}; zero-padded DEV acc={acc_Z:.16f}", flush=True)
    print(f"Q2={d_acc:.6f} ({c_cls['Q2']}) Q3={frac_changed:.6f} ({c_cls['Q3']}) "
          f"Q4={med_rel:.6f} ({c_cls['Q4']})", flush=True)
    print(f"Outcome: {outcome}. Written: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
