"""POST HOC DIAGNOSTIC for CONTRACT-006 V2. Not a contract gate.

It does not replace or change the CONTRACT-006 verdict: V2 (1.33e-1 against 1e-2) stays as registered.

Question: is the V2 failure dominated by the quadrature of the Rayleigh-Sommerfeld reference at source
pitch dx/2 for the oscillatory tilted input (about 3.75 source samples per wavelength), or by the
angular spectrum?

Same tapered tilted wave I2 (plus the untapered one as an optional extra), same coarse grid N = 128,
pitch DX = 0.4 mm, distance D = 18 mm, LAM = 0.75 mm as in experiments/run_rayleigh_sommerfeld.py.
That module is imported and not modified. Fields on the coarse observation grid:
  (a) ASM: optlab.propagation.angular_spectrum, the same call as CONTRACT-006
  (b) RS with source pitch DX/2 (CONTRACT-006 reference), 256 x 256 sources
  (c) RS with source pitch DX/4 (diagnostic reference), 512 x 512 sources over the same window

Metric: relative L2 complex-field error ||u_A - u_B|| / ||u_B||. The second-named field is the reference.

Usage:
  python -B experiments/diagnose_rs_quadrature.py

Output record (scratch, outside C:):
  D:/PROJECTS/.cognition/neuro3d-lab/tmp-diag/diagnose_rs_quadrature.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_rayleigh_sommerfeld as rs  # noqa: E402  (reused, not modified)

OUT_DIR = Path(rs.ROOT).parent / ".cognition" / "neuro3d-lab" / "tmp-diag"
V2_CONTRACT = 1.33e-1      # V2 value registered in CONTRACT-006 (ASM vs RS dx/2, tapered I2)
V2_THRESHOLD = 1e-2
SMALL = 1e-2               # "small" for (a) against (c), as stated in the diagnostic task


def run_diagnostic():
    t_start = time.time()
    N, DX, D, LAM = rs.N, rs.DX, rs.D, rs.LAM

    print("POST HOC DIAGNOSTIC for CONTRACT-006 V2. NOT a contract gate.")
    print("It does not replace or change the CONTRACT-006 verdict (V2 remains as registered).")
    print(f"grid N={N}, dx={DX * 1e3:.3f} mm, d={D * 1e3:.2f} mm, lambda={LAM * 1e3:.3f} mm, "
          f"sin_theta_b={rs.SIN_T:.6f}, taper rho0={rs.RHO_TAPER * 1e3:.1f} mm, order {rs.TAPER_ORDER}")
    print(f"ASM validity D < N*DX^2/LAM = {N * DX * DX / LAM * 1e3:.2f} mm: "
          f"{'yes' if D < N * DX * DX / LAM else 'NO'}")
    print(f"sources: dx/2 grid {2 * N}x{2 * N} ({(2 * N) ** 2} points); "
          f"dx/4 grid {4 * N}x{4 * N} ({(4 * N) ** 2} points)")
    print(f"RS chunk at dx/4: {max(1, rs.MAX_KERNEL_ENTRIES // (4 * N) ** 2)} observation points "
          f"(budget {rs.MAX_KERNEL_ENTRIES} kernel entries)")

    Xc, Yc = rs.grid(N, DX)
    Xb, Yb = rs.grid(2 * N, DX / 2)
    Xq, Yq = rs.grid(4 * N, DX / 4)

    t_c, u_c = rs.tilt_fields(Xc, Yc)     # tapered (I2), untapered (optional)
    t_b, u_b = rs.tilt_fields(Xb, Yb)
    t_q, u_q = rs.tilt_fields(Xq, Yq)
    F_b = np.stack([t_b.ravel(), u_b.ravel()], axis=1)
    F_q = np.stack([t_q.ravel(), u_q.ravel()], axis=1)

    # (a) ASM, same call as run_rayleigh_sommerfeld.run_full (stack order: tapered, untapered)
    t0 = time.time()
    asm = rs.angular_spectrum(np.stack([t_c, u_c]), DX, LAM, D)
    time_a = time.time() - t0
    print(f"(a) ASM done in {time_a:.2f} s")

    # (b) RS at source pitch dx/2 (CONTRACT-006 reference)
    t0 = time.time()
    rs_b = rs.rs_apply(Xc, Yc, Xb, Yb, (DX / 2) ** 2, D, F_b, kinds=("exact",))["exact"]
    time_b = time.time() - t0
    rs_b = rs_b.reshape(N, N, 2)
    print(f"(b) RS dx/2 done in {time_b:.2f} s")

    # (c) RS at source pitch dx/4 (diagnostic reference)
    t0 = time.time()
    rs_c = rs.rs_apply(Xc, Yc, Xq, Yq, (DX / 4) ** 2, D, F_q, kinds=("exact",))["exact"]
    time_c = time.time() - t0
    rs_c = rs_c.reshape(N, N, 2)
    print(f"(c) RS dx/4 done in {time_c:.2f} s")

    cases = {
        "tapered_I2_V2": (asm[0], rs_b[..., 0], rs_c[..., 0]),
        "untapered_I2_optional": (asm[1], rs_b[..., 1], rs_c[..., 1]),
    }
    errors = {}
    for name, (a, b, c) in cases.items():
        errors[name] = {
            "a_vs_b": rs.rel_l2(a, b),   # ASM against RS dx/2 (reference b); the CONTRACT-006 V2 metric
            "a_vs_c": rs.rel_l2(a, c),   # ASM against RS dx/4 (reference c)
            "b_vs_c": rs.rel_l2(b, c),   # RS dx/2 against RS dx/4 (reference c)
        }

    print()
    print("Relative L2 complex-field errors (reference = second-named field):")
    for name, er in errors.items():
        print(f"  {name}:")
        print(f"    (a) ASM vs (b) RS dx/2          = {er['a_vs_b']:.4e}")
        print(f"    (a) ASM vs (c) RS dx/4          = {er['a_vs_c']:.4e}")
        print(f"    (b) RS dx/2 vs (c) RS dx/4      = {er['b_vs_c']:.4e}")

    # Reproduction check against the fields stored by the registered CONTRACT-006 full run
    repro = {"stored_file_found": False}
    stored = Path(rs.OUT_LOCAL) / "rayleigh_sommerfeld_fields.npz"
    if stored.exists():
        with np.load(stored) as z:
            asm_s, rs2_s = z["asm"], z["rs_dx2"]   # stack/column order [gauss, tapered, digit, untapered]
            repro = {
                "stored_file_found": True,
                "asm_tapered_rel_diff": rs.rel_l2(asm[0], asm_s[1]),
                "rs_dx2_tapered_rel_diff": rs.rel_l2(rs_b[..., 0], rs2_s[..., 1]),
                "asm_untapered_rel_diff": rs.rel_l2(asm[1], asm_s[3]),
                "rs_dx2_untapered_rel_diff": rs.rel_l2(rs_b[..., 1], rs2_s[..., 3]),
            }
        print()
        print("Reproduction vs stored CONTRACT-006 fields (relative differences, expect ~1e-12 or less):")
        for key, val in repro.items():
            if key != "stored_file_found":
                print(f"  {key} = {val:.3e}")
    else:
        print(f"Reproduction check skipped: {stored} not found")

    e = errors["tapered_I2_V2"]
    e_ab, e_ac, e_bc = e["a_vs_b"], e["a_vs_c"], e["b_vs_c"]
    v2_reproduced = abs(e_ab - V2_CONTRACT) < 5e-4
    print()
    print(f"V2 recomputed (a) vs (b) = {e_ab:.4e}; registered V2 = {V2_CONTRACT:.2e}; "
          f"same to 3 digits: {v2_reproduced}")

    ref_dominated = 0.5 * V2_CONTRACT <= e_bc <= 2.0 * V2_CONTRACT   # "similar to 0.133" (within factor 2)
    asm_agrees = e_ac < SMALL
    print()
    print("Interpretation rule (tapered I2):")
    print(f"  rule 1, (b) vs (c) similar to V2 (within factor 2 of {V2_CONTRACT:.3f}): "
          f"{ref_dominated}  [{e_bc:.4e}]")
    print(f"  rule 2, (a) vs (c) below {SMALL:.0e}: {asm_agrees}  [{e_ac:.4e}]")
    if ref_dominated and asm_agrees:
        verdict = "Both rules hold: the V2 failure is attributable to the RS reference at dx/2."
    elif ref_dominated:
        verdict = ("Rule 1 holds, rule 2 does not: the dx/2 reference is far from dx/4, and the ASM is also "
                   ">= 0.01 from dx/4, so both contribute.")
    elif asm_agrees:
        verdict = ("Rule 2 holds, rule 1 does not: inconsistent with the registered V2 failure, "
                   "check the reproduction numbers.")
    else:
        verdict = ("Neither rule holds: the dx/2 reference is not far from dx/4 by a V2-sized amount, "
                   "and the ASM is not within 0.01 of the dx/4 reference.")
    print(f"  verdict: {verdict}")

    total = time.time() - t_start
    record = {
        "post_hoc": True,
        "label": "POST HOC DIAGNOSTIC for CONTRACT-006 V2; not a gate; does not change the CONTRACT-006 verdict",
        "parameters": {"lambda_m": LAM, "dx_m": DX, "N": N, "d_m": D, "sin_theta_b": rs.SIN_T,
                       "rho_taper_m": rs.RHO_TAPER, "taper_order": rs.TAPER_ORDER,
                       "source_pitch_b_m": DX / 2, "source_pitch_c_m": DX / 4,
                       "max_kernel_entries_per_chunk": rs.MAX_KERNEL_ENTRIES},
        "errors": errors,
        "reproduction_vs_stored_fields": repro,
        "v2_registered": V2_CONTRACT,
        "v2_recomputed_a_vs_b": e_ab,
        "rule_1_ref_dominated": ref_dominated,
        "rule_2_asm_below_0p01_vs_c": asm_agrees,
        "verdict": verdict,
        "timing_s": {"asm": time_a, "rs_dx2_two_fields": time_b, "rs_dx4_two_fields": time_c,
                     "total": total},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "diagnose_rs_quadrature.json"
    out_json.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print()
    print(f"record written to {out_json}")
    print(f"total runtime = {total:.1f} s")
    return 0


if __name__ == "__main__":
    code = run_diagnostic()
    print(f"exit code = {code}")
    sys.exit(code)
