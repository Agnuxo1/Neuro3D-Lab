"""CONTRATO-005 (POST HOC): robustness of the final CONTRATO-003 D2NN configuration at N = 256.

POST HOC. The CONTRATO-003 main results were known when CONTRATO-005 was written. This script
is a robustness check, not a confirmatory test. Contract: docs/CONTRACTS/CONTRATO-005-robustez-n256.md.

The model, the training loop, the split and the input encoding are imported from
experiments/run_d2nn_digits.py, which is not modified. Only the grid changes (N = 256 instead
of 128) and, with it, the padding of the input. The pixel size, wavelength, distances,
detector geometry (in physical units) and hyperparameters are the same.

Usage:
    python experiments/run_d2nn_robustness_n256.py --dry-run
        Builds the model, times 20 loss_and_grad calls and prints an estimate. Does not train
        and writes no files.
    python experiments/run_d2nn_robustness_n256.py
        Full run: one training run at N = 256 (seed 0), TEST accuracy and confusion matrix.

Outputs, written only by main():
    D:/PROJECTS/.cognition/neuro3d-lab/results/d2nn_robustness_n256.json   (full, not versioned)
    experiments/results/d2nn_robustness_n256_summary.json                  (versioned, no per-sample arrays)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

EXPERIMENTS = Path(__file__).resolve().parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT / "src"))

import run_d2nn_digits as base  # noqa: E402  (no side effects on import: main() is guarded there)
from optlab.d2nn import DiffractiveNetwork  # noqa: E402
from optlab.elements import detector_masks  # noqa: E402

# ---- fixed by CONTRATO-005 (post hoc) and inherited from CONTRATO-003 ----
N = 256                        # robustness grid; CONTRATO-003 used base.N = 128
DX = base.DX                   # 0.4 mm, unchanged: only the window grows
LAM = base.LAM                 # 0.75 mm
DIST = base.DIST               # [18 mm] * 4
PAD = (N - 64) // 2            # 96 px per side: the 64 x 64 digit is centred on 256 x 256
SCALE = 100.0                  # beta selected on DEV in CONTRATO-003 (checked against its summary)
LR = 0.02                      # lr selected on DEV in CONTRATO-003 (checked against its summary)
EPOCHS = base.EPOCHS           # 30
BATCH = base.BATCH             # 32
MODEL_SEED = 0
TRAIN_RNG_SEED = 0
H1_THRESHOLD = 0.85            # stability also requires TEST_256 >= 0.85
STABILITY_TOL = 0.01           # stability requires |TEST_256 - TEST_128| <= 0.01
N_CLASSES = 10
REGION_PX = 12
N_TEST_EXPECTED = 360

REF_SUMMARY = EXPERIMENTS / "results" / "d2nn_digits_summary.json"
REF_FULL = Path("D:/PROJECTS/.cognition/neuro3d-lab/results/d2nn_digits.json")
OUT_LOCAL = Path("D:/PROJECTS/.cognition/neuro3d-lab/results")
OUT_FULL = OUT_LOCAL / "d2nn_robustness_n256.json"
LOG = OUT_LOCAL / "d2nn_robustness_n256.log"
OUT_SUMMARY = EXPERIMENTS / "results" / "d2nn_robustness_n256_summary.json"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def regions_physical():
    """Ten detector regions in physical units (offsets in multiples of dx, so any grid size)."""
    cols = [-40, -20, 0, 20, 40]    # +x to the right
    rows = [20, -20]                 # +y up; row 0 is the top
    return [{"cx_m": cols[k % 5] * DX, "cy_m": rows[k // 5] * DX,
             "w_m": REGION_PX * DX, "h_m": REGION_PX * DX} for k in range(N_CLASSES)]


def build_masks(n):
    return detector_masks(n, DX, regions_physical())


def check_geometry():
    """The N = 128 masks must equal the imported ones; each N = 256 region must hold 12 x 12 px."""
    if not np.array_equal(build_masks(base.N), base.build_masks()):
        raise RuntimeError("N = 128 mask geometry differs from run_d2nn_digits.build_masks()")
    masks = build_masks(N)
    counts = masks.reshape(len(masks), -1).sum(axis=1).astype(int).tolist()
    if counts != [REGION_PX * REGION_PX] * N_CLASSES:
        raise RuntimeError(f"unexpected region sizes at N = {N}: {counts}")
    return masks, counts


def validity_limit_m():
    return N * DX**2 / LAM


def check_validity():
    limit = validity_limit_m()
    if not max(DIST) < limit:
        raise RuntimeError(f"distance {max(DIST)} m is outside the validity limit {limit} m")
    return limit


def encode_field_n256(img8_one):
    """Same logic as base.encode_field (upsample, unit energy, zero-pad), with PAD px per side."""
    return np.pad(base.encode64(img8_one), PAD).astype(complex)


class LazyFields:
    """Sequence of N = 256 input fields, encoded on access.

    One field is 1 MiB (complex128). Holding all split fields in memory would need about 1.5 GB.
    Encoding is deterministic, so encoding on access does not change any value.
    """

    def __init__(self, img8, ids):
        self._img8 = img8
        self._ids = np.asarray(ids)

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, i):
        return encode_field_n256(self._img8[self._ids[i]])

    def __iter__(self):
        for i in range(len(self._ids)):
            yield self[i]


def load_reference():
    """CONTRATO-003 seed-0 TEST accuracy and confusion matrix, read from its files (nothing typed by hand)."""
    summ = json.loads(REF_SUMMARY.read_text(encoding="utf-8"))
    full = json.loads(REF_FULL.read_text(encoding="utf-8"))
    sel = summ["selected"]
    if sel["beta"] != SCALE or sel["lr"] != LR:
        raise RuntimeError(f"configuration mismatch: CONTRATO-003 selected beta={sel['beta']} lr={sel['lr']}")
    seed0 = next(f for f in summ["final"] if f["seed"] == 0)
    cm = full["confusion_matrix_seed0"]
    total = sum(sum(row) for row in cm)
    correct = sum(cm[i][i] for i in range(N_CLASSES))
    if total != summ["sizes"]["test"] or total != N_TEST_EXPECTED:
        raise RuntimeError(f"unexpected TEST size in the reference: {total}")
    if abs(correct / total - seed0["test_acc"]) > 1e-12:
        raise RuntimeError("reference confusion matrix does not match the reference seed-0 TEST accuracy")
    return {
        "beta": sel["beta"], "lr": sel["lr"],
        "test_acc_seed0": float(seed0["test_acc"]),
        "test_acc_seeds": [float(f["test_acc"]) for f in summ["final"]],
        "confusion_matrix_seed0": cm,
    }


def dry_run():
    img8, y_all, tr, _dv, te = base.load_split()
    masks, counts = check_geometry()
    limit = check_validity()
    ref = load_reference()
    model = DiffractiveNetwork(N, DX, LAM, DIST, masks, scale=SCALE, seed=MODEL_SEED)
    u = encode_field_n256(img8[tr[0]])
    label = int(y_all[tr[0]])

    n_calls = 20
    t0 = time.perf_counter()
    for _ in range(n_calls):
        model.loss_and_grad(u, label)
    per_call = (time.perf_counter() - t0) / n_calls

    train_s = per_call * EPOCHS * len(tr)
    pred_s = per_call * len(te)        # upper bound: a forward pass costs less than loss_and_grad
    total_s = train_s + pred_s

    print("CONTRATO-005 (POST HOC) dry run: no training, no files written")
    print(f"grid N = {N}, dx = {DX * 1e3:.1f} mm, window = {N * DX * 1e3:.1f} mm, "
          f"validity limit = {limit * 1e3:.1f} mm, distance = {max(DIST) * 1e3:.0f} mm (inside)")
    print(f"detector regions at N = {N}: {counts} px each; N = 128 geometry check against "
          f"run_d2nn_digits.build_masks(): passed")
    print(f"reference CONTRATO-003 N = 128, seed 0: TEST = {ref['test_acc_seed0']:.6f} "
          f"(beta = {ref['beta']}, lr = {ref['lr']})")
    print(f"loss_and_grad at N = {N}: {per_call * 1e3:.1f} ms per call (mean of {n_calls} calls)")
    print(f"estimated training time: {train_s / 60:.1f} min "
          f"({EPOCHS} epochs x {len(tr)} samples = {EPOCHS * len(tr)} calls)")
    print(f"estimated total incl. TEST prediction (upper bound): {total_s / 60:.1f} min "
          f"({total_s / 3600:.2f} h)")
    return 0


def main():
    t0 = time.time()
    check_validity()
    ref = load_reference()
    masks, counts = check_geometry()
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    log("CONTRATO-005 POST HOC robustness check: a robustness check, not a confirmatory test")

    img8, y_all, tr, _dv, te = base.load_split()
    log(f"split (seed {base.SPLIT_SEED}): TRAIN={len(tr)} TEST={len(te)}; DEV not used at N = {N}")
    y_tr = y_all[tr]
    y_te = y_all[te]
    U_tr = LazyFields(img8, tr)
    U_te = LazyFields(img8, te)

    model = DiffractiveNetwork(N, DX, LAM, DIST, masks, scale=SCALE, seed=MODEL_SEED)
    log(f"training N = {N}, beta = {SCALE}, lr = {LR}, model seed = {MODEL_SEED}, "
        f"epochs = {EPOCHS}, batch = {BATCH}")
    base.train(model, U_tr, y_tr, LR, np.random.default_rng(TRAIN_RNG_SEED), epochs=EPOCHS, batch=BATCH)

    pred_te = base.predictions(model, U_te)
    test_256 = base.acc(pred_te, y_te)
    n_correct_256 = int(np.sum(pred_te == y_te))
    cm_256 = confusion_matrix(y_te, pred_te, labels=list(range(N_CLASSES))).tolist()
    test_128 = ref["test_acc_seed0"]
    diff = float(test_256 - test_128)
    within_tol = abs(diff) <= STABILITY_TOL
    above_h1 = test_256 >= H1_THRESHOLD
    verdict = "stable" if (within_tol and above_h1) else "unstable"
    elapsed = round(time.time() - t0, 1)

    criterion = {
        "rule": "stable if |TEST_256 - TEST_128| <= 0.01 and TEST_256 >= 0.85; unstable otherwise",
        "tolerance": STABILITY_TOL, "h1_threshold": H1_THRESHOLD,
        "within_tolerance": bool(within_tol), "above_h1_threshold": bool(above_h1),
    }
    grid = {"N_reference": base.N, "N": N, "dx_m": DX, "lambda_m": LAM, "distances_m": DIST,
            "window_m": N * DX, "validity_limit_m": validity_limit_m(), "pad_px_per_side": PAD,
            "region_px": REGION_PX, "region_sizes_px": counts}
    hyper = {"beta": SCALE, "lr": LR, "epochs": EPOCHS, "batch": BATCH, "model_seed": MODEL_SEED,
             "train_rng_seed": TRAIN_RNG_SEED, "split_seed": base.SPLIT_SEED,
             "selected_in": "CONTRATO-003 (DEV, seed 0)"}
    reference = {"source": str(REF_SUMMARY), "test_acc_seed0": test_128,
                 "test_acc_seeds": ref["test_acc_seeds"], "confusion_matrix_seed0": ref["confusion_matrix_seed0"]}
    n256 = {"test_acc_seed0": test_256, "n_correct_seed0": n_correct_256, "n_test": int(len(te)),
            "confusion_matrix_seed0": cm_256}
    full = {
        "contract": "CONTRATO-005", "post_hoc": True,
        "statement": "POST HOC robustness check. CONTRATO-003 main results were known when this was "
                     "written. Retraining at N = 256 is a new trajectory; results need not match exactly.",
        "criterion": criterion, "grid": grid, "hyperparameters": hyper,
        "reference_n128": reference, "n256": n256,
        "difference_256_minus_128": diff, "verdict": verdict,
        "per_sample": {"y_test": [int(v) for v in y_te], "pred_test_seed0_n256": [int(p) for p in pred_te]},
        "elapsed_s": elapsed,
    }
    OUT_FULL.write_text(json.dumps(full, indent=1), encoding="utf-8")

    summary = {
        "contract": "CONTRATO-005", "post_hoc": True,
        "statement": full["statement"],
        "criterion": criterion, "grid": grid, "hyperparameters": hyper,
        "test_128_seed0": test_128, "test_256_seed0": test_256,
        "n_correct_128_seed0": int(round(test_128 * len(te))), "n_correct_256_seed0": n_correct_256,
        "n_test": int(len(te)),
        "difference_256_minus_128": diff,
        "verdict": verdict,
        "confusion_matrix_seed0_n256": cm_256,
        "confusion_matrix_seed0_n128": ref["confusion_matrix_seed0"],
        "confusion_matrix_note": "rows = true class, columns = predicted class (sklearn convention)",
        "reference_n128_test_seeds": ref["test_acc_seeds"],
        "elapsed_s": elapsed,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    log(f"TEST seed 0: N = 128 {test_128:.6f} | N = 256 {test_256:.6f} | difference {diff:+.6f}")
    log(f"VERDICT (post hoc): {verdict}")
    log(f"fin en {elapsed:.0f} s")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CONTRATO-005 (post hoc): N = 256 robustness of the CONTRATO-003 D2NN.")
    p.add_argument("--dry-run", action="store_true",
                   help="build the model and time 20 gradient calls; no training, no files written")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(dry_run() if args.dry_run else main())
