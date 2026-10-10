"""POST HOC, DEV only: aperture convention (A) for the seed-0 D2NN DEV predictions (C12b).

Pre-registered in docs/CONTRACTS/CONTRATO-011-aperture-dev.md (written before any run).
Field on a 256 grid (input centred at offset 64). After each propagation the field is multiplied
by the 128 x 128 central window mask: light leaving the window is lost. TEST is not encoded,
not loaded for evaluation and not evaluated. optlab is not modified.

Usage: python -B experiments/c12b_aperture_dev_d2nn.py
Input: experiments/results/d2nn_seed0_phases.npz (this script does not train).
Output: experiments/results/c12b_aperture_dev_d2nn.json, written only if Q1 reproduces.
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
OUT_PATH = EXPERIMENTS / "results" / "c12b_aperture_dev_d2nn.json"

BETA = 100.0
SEED = 0
N = R.N                       # 128: registered grid, periodic window 51.2 mm
NZ = 256                      # zero-padded grid, window 102.4 mm
OFF = (NZ - N) // 2           # 64: offset of the 128 window inside the 256 grid
L = len(R.DIST) - 1           # 3 phase layers
N_DEV_EXPECTED = 359

# ---- constants fixed in CONTRATO-010 / CONTRATO-011 before any run ----
EXPECTED_DEV_ACC = 0.9554317548746518    # P, periodic
Z_DEV_ACC = 0.8495821727019499           # Z, zero-padded with transparent outside (CONTRATO-010)
Q1_TOL = 1e-9
RULE_TOL = 0.01


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


def window_mask():
    m = np.zeros((NZ, NZ), dtype=float)
    m[OFF:OFF + N, OFF:OFF + N] = 1.0
    return m


def padded_phase_exp(phases):
    """exp(i phi) inside the central 128 window, 1 elsewhere (the field is zero there anyway)."""
    ph = np.zeros((L, NZ, NZ), dtype=float)
    ph[:, OFF:OFF + N, OFF:OFF + N] = phases
    return np.exp(1j * ph)


def forward_aperture(u128, ph_exp256, win, masks128):
    a = np.zeros((NZ, NZ), dtype=complex)
    a[OFF:OFF + N, OFF:OFF + N] = u128                      # input centred in the 256 grid
    for k in range(L):
        a = angular_spectrum(a, R.DX, R.LAM, R.DIST[k]) * win   # light outside the window is lost
        a = a * ph_exp256[k]
    f = angular_spectrum(a, R.DX, R.LAM, R.DIST[L]) * win
    I_central = np.abs(f[OFF:OFF + N, OFF:OFF + N]) ** 2
    return BETA * np.tensordot(masks128, I_central, axes=([1, 2], [0, 1]))


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

    # ---- A: aperture propagation ----
    ph_exp256 = padded_phase_exp(phases)
    win = window_mask()
    s_A = np.array([forward_aperture(u, ph_exp256, win, masks128) for u in U_dv])
    pred_A = np.argmax(s_A, axis=1)
    acc_A = R.acc(pred_A, y_dv)

    # ---- pre-registered rule ----
    d_AP = float(abs(acc_A - EXPECTED_DEV_ACC))
    d_AZ = float(abs(acc_A - Z_DEV_ACC))
    r1 = d_AP <= RULE_TOL
    r2 = d_AZ <= RULE_TOL
    if r1 and r2:
        outcome = "R1 and R2 both satisfied (ambiguous; report as inconclusive)"
    elif r1:
        outcome = "R1"
    elif r2:
        outcome = "R2"
    else:
        outcome = "R3"

    frac_PA = float(np.mean(pred_P != pred_A))
    result = {
        "contract": "CONTRATO-011",
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
        "aperture": {"grid": NZ, "offset": OFF, "dev_acc": float(acc_A)},
        "reference": {"P": EXPECTED_DEV_ACC, "Z": Z_DEV_ACC},
        "abs_diff_A_P": d_AP,
        "abs_diff_A_Z": d_AZ,
        "rule_tolerance": RULE_TOL,
        "fraction_changed_P_vs_A": frac_PA,
        "n_changed_P_vs_A": int(np.sum(pred_P != pred_A)),
        "fraction_changed_Z_vs_A": None,
        "outcome": outcome,
        "elapsed_s": round(time.time() - t0, 1),
    }
    # Z predictions are recomputed (transparent outside the window) to compare per image
    ph_exp_z = ph_exp256
    def fwd_z(u):
        a = np.zeros((NZ, NZ), dtype=complex)
        a[OFF:OFF + N, OFF:OFF + N] = u
        for k in range(L):
            a = angular_spectrum(a, R.DX, R.LAM, R.DIST[k]) * ph_exp_z[k]
        f = angular_spectrum(a, R.DX, R.LAM, R.DIST[L])
        return BETA * np.tensordot(masks128, np.abs(f[OFF:OFF + N, OFF:OFF + N]) ** 2,
                                   axes=([1, 2], [0, 1]))
    pred_Z = np.argmax(np.array([fwd_z(u) for u in U_dv]), axis=1)
    acc_Z = R.acc(pred_Z, y_dv)
    result["zero_padded_recomputed_dev_acc"] = float(acc_Z)
    result["zero_padded_matches_CONTRATO_010"] = bool(abs(acc_Z - Z_DEV_ACC) <= Q1_TOL)
    result["fraction_changed_Z_vs_A"] = float(np.mean(pred_Z != pred_A))
    result["n_changed_Z_vs_A"] = int(np.sum(pred_Z != pred_A))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=1) + chr(10), encoding="utf-8")
    print(f"Q1 ok: periodic DEV acc={acc_P:.16f}; aperture DEV acc={acc_A:.16f}", flush=True)
    print(f"|A-P|={d_AP:.6f} |A-Z|={d_AZ:.6f} changed P/A={frac_PA:.6f} "
          f"Z/A={result['fraction_changed_Z_vs_A']:.6f}", flush=True)
    print(f"Outcome: {outcome}. Written: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
