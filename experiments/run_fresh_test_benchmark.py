"""CONTRACT-007: fresh-test benchmark of a 3-layer phase-only D2NN against three baselines.

Protocol, thresholds and grids: docs/CONTRACTS/CONTRATO-007-test-fresco.md. This file
implements that protocol. It does not change any threshold or grid.

TRAIN and DEV are the CONTRACT-003 split of sklearn load_digits (seed 2026). TEST_FRESH is
the set of UCI optdigits training-file samples that are not among the 1797 load_digits
samples (3823 samples, never used by a previous analysis). TEST_FRESH is evaluated once per
configuration and seed, and only after every selection on DEV has been made. The 360-sample
TEST split of CONTRACT-003 is counted but never evaluated.

Usage:
  python experiments/run_fresh_test_benchmark.py --dry-run
      Checks the data counts, times 20 D2NN gradient calls, 20 D2NN forward passes, one
      logistic fit, one MLP fit and one CNN epoch, and prints estimates. It writes no
      results. The only file it may write is the OpenML download cache under
      $NEURO3D_LAB_OUT/cache/openml.
  python experiments/run_fresh_test_benchmark.py --run
      Full protocol. CPU only, several hours. Not executed when this file was written.
  python experiments/run_fresh_test_benchmark.py --part PART
      One part of the same protocol, for running in parallel. PART is one of
        dev:K     K = 0..2, DEV grid point beta = BETA_GRID[K], lr 0.02, seed 0
        seed:S    S = 0..9, selected beta, model seed S and rng seed S; DEV and TEST_FRESH
        controls  C1 label-permuted seeds 0..2 and C2 untrained seeds 0..9
        baselines (a) to (d), each with its DEV selection; TEST_FRESH predictions saved
        assemble  reads every part file, refuses if one is missing, writes the outputs
      Each part writes $NEURO3D_LAB_OUT/results/fresh_parts/<part>.json, with ':' replaced by
      '_' (dev_0.json, seed_3.json, controls.json, baselines.json). seed:S and controls read the
      selected beta from the three dev parts and refuse to run without them. A part refuses to
      overwrite its file, so TEST_FRESH is evaluated once per configuration and seed.

Outputs of --run (and of --part assemble, which writes the same files):
  experiments/results/fresh_test_benchmark_summary.json   versioned, no per-sample arrays
  $NEURO3D_LAB_OUT/results/fresh_test_benchmark.json      full record with per-sample predictions
  $NEURO3D_LAB_OUT/results/fresh_test_benchmark.log       progress log (--run and assemble)
NEURO3D_LAB_OUT defaults to D:/PROJECTS/.cognition/neuro3d-lab. Nothing is written to C:.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

# CPU only by contract: hide CUDA before torch is imported. The GPU queue is not used.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.datasets import fetch_openml, load_digits  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlab.d2nn import DiffractiveNetwork  # noqa: E402
from optlab.elements import detector_masks  # noqa: E402
from optlab.stats import wilson_interval  # noqa: E402

# ---- fixed by CONTRACT-007 (do not edit after the full run) ----
SPLIT_SEED = 2026
N = 128
DX = 0.4e-3
LAM = 0.75e-3
DIST = [18e-3] * 4                # entrance -> layer 1 -> layer 2 -> layer 3 -> detector
BETA_GRID = [100.0, 200.0, 400.0]
LR_D2NN = 0.02
EPOCHS = 30
BATCH = 32
D2NN_SEEDS = list(range(10))
C1_SEEDS = [0, 1, 2]
C2_SEEDS = list(range(10))
LOGIT_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MLP_HIDDEN = 653
MLP_ALPHA_GRID = [1e-4, 1e-3, 1e-2]
CNN_EPOCHS = 60
CNN_LR = 1e-3
CNN_BATCH = 32
OPENML_NAME = "optdigits"
OPENML_VERSION = 1
OPENML_ROWS = 5620
SOURCE_ROWS = 1797
FRESH_ROWS = 3823
TRAIN_ROWS = 1078
DEV_ROWS = 359
TEST_ROWS = 360                   # counted only; never evaluated
Z = 1.96

OUT_BASE = Path(os.environ.get("NEURO3D_LAB_OUT", ROOT.parent / ".cognition" / "neuro3d-lab"))
OUT_LOCAL = OUT_BASE / "results"
OUT_CACHE = OUT_BASE / "cache" / "openml"
OUT_REPO = ROOT / "experiments" / "results"
SUMMARY_PATH = OUT_REPO / "fresh_test_benchmark_summary.json"
FULL_PATH = OUT_LOCAL / "fresh_test_benchmark.json"
LOG_PATH = OUT_LOCAL / "fresh_test_benchmark.log"
PARTS_DIR = OUT_LOCAL / "fresh_parts"

DECISION_RECORD = {
    "consulted": "router.py jev; provenance=jev; status=connected; model=jev-1.13.0; remote_decision=true",
    "cnn_selection": {"jev": "fixed epoch 60, no DEV selection (confidence 0.73)",
                      "adopted": "epoch with best DEV accuracy among 60, seed 0, ties to earlier epoch",
                      "reason": "the protocol fixed before the consult requires DEV selection"},
    "mlp_selection": {"jev": "single alpha 1e-4, no selection (confidence 0.41)",
                      "adopted": "alpha in {1e-4, 1e-3, 1e-2} selected on DEV, seed 0",
                      "reason": "the protocol fixed before the consult requires DEV selection"},
    "best_baseline_tie_break": {"jev": "earlier in order (a) to (d) (confidence 0.19)",
                                "adopted": "more trainable parameters",
                                "reason": "low JEV confidence; this rule makes H2 harder to pass"},
    "c2_seeds": {"jev": "seeds 0 to 2, verdict on the mean (confidence 0.16)",
                 "adopted": "seeds 0 to 9, verdict on the mean, every seed reported",
                 "reason": "matches the 10 D2NN seeds; forward passes only, so cheap"},
}


# ---------------------------------------------------------------- utilities
_WRITE_LOG = False


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _WRITE_LOG:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:  # psutil is optional for the memory report
        return None


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def wilson(correct, n):
    lo, hi = wilson_interval(int(correct), int(n), z=Z)
    return [lo, hi]


def paired(pred_d2nn, pred_base, y):
    """b = D2NN correct and baseline wrong; c = baseline correct and D2NN wrong.
    Two-sided exact McNemar test (binomial test on the discordant pairs)."""
    cd = pred_d2nn == y
    cb = pred_base == y
    b = int(np.sum(cd & ~cb))
    c = int(np.sum(~cd & cb))
    p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c > 0 else 1.0
    return {"d2nn_only_correct": b, "baseline_only_correct": c, "p_two_sided_exact": p}


def confusion(y, p):
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, p), 1)
    return cm.tolist()


# ---------------------------------------------------------------- data
def load_split(d):
    """CONTRACT-003 split logic, copied from experiments/run_d2nn_digits.py (not imported)."""
    img8 = d.images / 16.0
    y = d.target.astype(int)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=SPLIT_SEED)
    dv, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SPLIT_SEED)
    return img8, y, tr, dv, te


def sample_key(row_u8, label):
    """Exact identity of one sample: its 64 integer pixel values and its label."""
    return hashlib.sha256(row_u8.tobytes() + bytes([int(label)])).hexdigest()


def load_fresh(d):
    """OpenML optdigits v1 rows that are not an exact match (pixels and label) of a
    load_digits sample. Multiset matching, so duplicated rows are handled correctly."""
    src = np.asarray(d.data, dtype=float)
    if not np.array_equal(src, np.rint(src)):
        raise ValueError("load_digits pixels are not integers")
    src_u8 = np.rint(src).astype(np.uint8)
    src_y = d.target.astype(int)
    pool = Counter(sample_key(src_u8[i], src_y[i]) for i in range(len(src_y)))

    op = fetch_openml(OPENML_NAME, version=OPENML_VERSION, as_frame=False,
                      parser="liac-arff", data_home=str(OUT_CACHE))
    X_op = np.asarray(op.data, dtype=float)
    y_op = np.asarray(op.target).astype(int)
    if X_op.shape != (OPENML_ROWS, 64):
        raise ValueError(f"unexpected OpenML shape {X_op.shape}")
    if not np.array_equal(X_op, np.rint(X_op)) or X_op.min() < 0 or X_op.max() > 16:
        raise ValueError("OpenML pixels are outside the integer range 0..16")
    op_u8 = X_op.astype(np.uint8)

    in_source = np.zeros(OPENML_ROWS, dtype=bool)
    for i in range(OPENML_ROWS):
        k = sample_key(op_u8[i], y_op[i])
        if pool.get(k, 0) > 0:
            pool[k] -= 1
            in_source[i] = True
    n_matched = int(in_source.sum())
    if n_matched != SOURCE_ROWS:
        raise ValueError(f"exact matching found {n_matched} of {SOURCE_ROWS} source samples")
    fresh_rows = np.flatnonzero(~in_source)
    if len(fresh_rows) != FRESH_ROWS:
        raise ValueError(f"expected {FRESH_ROWS} fresh samples, got {len(fresh_rows)}")
    # 8x8 like load_digits.images (row-major, same as the matching above), so that encode64,
    # to_tensor and the D2NN encoder see the same layout for TEST_FRESH as for TRAIN and DEV.
    return (X_op[fresh_rows] / 16.0).reshape(-1, 8, 8), y_op[fresh_rows], fresh_rows, n_matched


def prepare():
    d = load_digits()
    img8, y_src, tr, dv, te = load_split(d)
    sizes = (len(y_src), len(tr), len(dv), len(te))
    if sizes != (SOURCE_ROWS, TRAIN_ROWS, DEV_ROWS, TEST_ROWS):
        raise ValueError(f"split sizes differ from CONTRACT-003: {sizes}")
    X_f, y_f, fresh_rows, n_matched = load_fresh(d)
    counts = {
        "load_digits_source": int(len(y_src)),
        "train_contract003": int(len(tr)),
        "dev_contract003": int(len(dv)),
        "test_contract003_counted_not_evaluated": int(len(te)),
        "openml_optdigits_v1_rows": OPENML_ROWS,
        "source_rows_matched_exactly": n_matched,
        "test_fresh_uci_training_file": int(len(y_f)),
    }
    return {
        "tr": (img8[tr], y_src[tr]),
        "dv": (img8[dv], y_src[dv]),
        "fresh": (X_f, y_f),
        "fresh_rows": fresh_rows,
        "counts": counts,
    }


# ---------------------------------------------------------------- encodings
def encode64(a8):
    a = np.kron(a8, np.ones((8, 8)))
    return a / np.linalg.norm(a)


def encode_field(a8):
    return np.pad(encode64(a8), 32).astype(complex)


def flat64(A8):
    return np.stack([encode64(a).ravel() for a in A8])


def raw64(A8):
    return A8.reshape(len(A8), 64)


def predict_chunked(predict, feat, A8, chunk=512):
    """Predict in chunks to bound memory (the 64x64 feature matrix is about 125 MB for TEST_FRESH)."""
    out = [predict(feat(A8[s:s + chunk])) for s in range(0, len(A8), chunk)]
    return np.concatenate(out).astype(int)


# ---------------------------------------------------------------- D2NN
def build_masks():
    cols = [-40, -20, 0, 20, 40]     # pixels, +x to the right
    rows = [20, -20]                 # pixels, +y up; row 0 is the top
    regions = [{"cx_m": cols[k % 5] * DX, "cy_m": rows[k // 5] * DX,
                "w_m": 12 * DX, "h_m": 12 * DX} for k in range(10)]
    return detector_masks(N, DX, regions)


def make_d2nn(masks, beta, seed):
    return DiffractiveNetwork(N, DX, LAM, DIST, masks, scale=beta, seed=seed)


def train_d2nn(model, A8, y, lr, rng):
    """Training loop copied from experiments/run_d2nn_digits.py: Adam, mini-batches of 32,
    mean analytic gradient per batch, one Adam step per batch."""
    m = np.zeros_like(model.phases)
    v = np.zeros_like(model.phases)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    idx = np.arange(len(y))
    for _ in range(EPOCHS):
        rng.shuffle(idx)
        for s in range(0, len(idx), BATCH):
            bi = idx[s:s + BATCH]
            g = np.zeros_like(model.phases)
            for i in bi:
                _, gi = model.loss_and_grad(encode_field(A8[i]), int(y[i]))
                g += gi
            g /= len(bi)
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            model.phases -= lr * mh / (np.sqrt(vh) + eps)
    return model


def d2nn_predict(model, A8):
    return np.array([model.predict(encode_field(a)) for a in A8], dtype=int)


# ---------------------------------------------------------------- baselines
def n_params_sklearn(clf):
    if hasattr(clf, "coefs_"):                       # MLP
        return int(sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_))
    return int(clf.coef_.size + clf.intercept_.size)  # multinomial logistic regression


def make_mlp(alpha):
    return MLPClassifier(hidden_layer_sizes=(MLP_HIDDEN,), activation="relu", solver="adam",
                         alpha=alpha, batch_size=BATCH, learning_rate_init=1e-3,
                         max_iter=200, random_state=0)


def make_cnn():
    # 1x8x8 -> conv 32 (3x3, pad 1) -> pool 2 -> 4x4x32 -> conv 112 (3x3, pad 1) -> 4x4x112
    # -> flatten 1792 -> linear 10. Parameters: 320 + 32368 + 17930 = 50618.
    return nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 112, kernel_size=3, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(112 * 4 * 4, 10))


def to_tensor(A8):
    return torch.as_tensor(np.asarray(A8, dtype=np.float32)[:, None, :, :])


def cnn_predict(net, X):
    net.eval()
    with torch.no_grad():
        return net(X).argmax(dim=1).numpy().astype(int)


def train_cnn(net, X, y, rng, dev=None, epochs=CNN_EPOCHS):
    """Adam (lr 1e-3, default betas), batch 32. With dev=(X_dev, y_dev), DEV accuracy is
    recorded after every epoch and the epoch with the highest DEV accuracy is kept
    (ties go to the earlier epoch). Returns (dev_curve, best) with the stored state."""
    opt = torch.optim.Adam(net.parameters(), lr=CNN_LR)
    lossf = nn.CrossEntropyLoss()
    Y = torch.as_tensor(y, dtype=torch.long)
    idx = np.arange(len(y))
    curve, best = [], None
    for ep in range(1, epochs + 1):
        net.train()
        rng.shuffle(idx)
        for s in range(0, len(idx), CNN_BATCH):
            bi = torch.as_tensor(idx[s:s + CNN_BATCH])
            opt.zero_grad()
            loss = lossf(net(X[bi]), Y[bi])
            loss.backward()
            opt.step()
        if dev is not None:
            a = float(np.mean(cnn_predict(net, dev[0]) == dev[1]))
            curve.append(a)
            if best is None or a > best["dev_acc"]:
                best = {"epoch": ep, "dev_acc": a, "state": copy.deepcopy(net.state_dict())}
    return curve, best


def select_logit(X_tr, y_tr, X_dv, y_dv):
    grid, best = [], None
    for C in LOGIT_C_GRID:                              # ascending: ties keep the smaller C
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=C, max_iter=3000).fit(X_tr, y_tr)
        a = float(np.mean(clf.predict(X_dv) == y_dv))
        grid.append({"C": C, "dev_acc": a, "n_iter": int(np.max(clf.n_iter_))})
        if best is None or a > best["dev_acc"]:
            best = {"C": C, "dev_acc": a, "model": clf, "n_params": n_params_sklearn(clf)}
    return grid, best


def select_mlp(X_tr, y_tr, X_dv, y_dv):
    grid, best = [], None
    for alpha in MLP_ALPHA_GRID:                        # ascending: ties keep the smaller alpha
        clf = make_mlp(alpha)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
        a = float(np.mean(clf.predict(X_dv) == y_dv))
        grid.append({"alpha": alpha, "dev_acc": a, "n_iter": int(clf.n_iter_)})
        if best is None or a > best["dev_acc"]:
            best = {"alpha": alpha, "dev_acc": a, "model": clf, "n_params": n_params_sklearn(clf)}
    return grid, best


# ---------------------------------------------------------------- dry run
def dry_run():
    data = prepare()
    c = data["counts"]
    A_tr, y_tr = data["tr"]
    A_dv, y_dv = data["dv"]
    A_f, y_f = data["fresh"]
    print("DATA COUNTS")
    for k, v in c.items():
        print(f"  {k}: {v}")
    print(f"  test_fresh per-class counts: {np.bincount(y_f, minlength=10).tolist()}")
    print(f"  rss_mb after loading: {rss_mb()}")

    masks = build_masks()
    n_cnn = sum(p.numel() for p in make_cnn().parameters())
    print("PARAMETERS")
    print(f"  D2NN phases (3 x {N} x {N}): {3 * N * N}")
    print(f"  (a) logistic 64x64: {64 * 64 * 10 + 10}")
    print(f"  (b) logistic 8x8 raw: {64 * 10 + 10}")
    print(f"  (c) MLP 64-{MLP_HIDDEN}-10: {64 * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN * 10 + 10}")
    print(f"  (d) CNN (torch module count): {n_cnn}")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(y_tr), size=21, replace=False)
    model = make_d2nn(masks, BETA_GRID[0], seed=0)
    model.loss_and_grad(encode_field(A_tr[idx[0]]), int(y_tr[idx[0]]))   # warm-up, not timed
    t = time.perf_counter()
    for i in idx[1:]:
        model.loss_and_grad(encode_field(A_tr[i]), int(y_tr[i]))
    t_grad = (time.perf_counter() - t) / 20

    t = time.perf_counter()
    for a in A_dv[:20]:
        model.predict(encode_field(a))
    t_fwd = (time.perf_counter() - t) / 20

    X_tr64 = flat64(A_tr)
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        LogisticRegression(C=1.0, max_iter=3000).fit(X_tr64, y_tr)
    t_lr = time.perf_counter() - t
    del X_tr64

    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        make_mlp(1e-4).fit(raw64(A_tr), y_tr)
    t_mlp = time.perf_counter() - t

    torch.manual_seed(0)
    net = make_cnn()
    t = time.perf_counter()
    train_cnn(net, to_tensor(A_tr), y_tr, np.random.default_rng(0), epochs=1)
    t_cnn_epoch = time.perf_counter() - t

    n_tr, n_dv, n_f = len(y_tr), len(y_dv), len(y_f)
    n_train_runs = len(BETA_GRID) + len(D2NN_SEEDS) + len(C1_SEEDS)
    grad_calls = n_train_runs * EPOCHS * n_tr
    fwd_calls = (len(BETA_GRID) * n_dv + len(D2NN_SEEDS) * (n_dv + n_f)
                 + (len(C1_SEEDS) + len(C2_SEEDS)) * n_f)
    est = {
        "d2nn_training_seconds": grad_calls * t_grad,
        "d2nn_forward_seconds": fwd_calls * t_fwd,
        "logistic_64x64_fits_seconds (6 fits at C=1 timing; larger C may need more iterations)": 6 * t_lr,
        "mlp_fits_seconds (3 fits)": 3 * t_mlp,
        "cnn_60_epochs_seconds": CNN_EPOCHS * t_cnn_epoch,
    }
    total = sum(est.values())
    print("TIMING (this machine)")
    print(f"  D2NN gradient call: {t_grad * 1e3:.2f} ms (20 calls)")
    print(f"  D2NN forward call: {t_fwd * 1e3:.2f} ms (20 calls)")
    print(f"  logistic fit C=1 (64x64): {t_lr:.2f} s")
    print(f"  MLP fit alpha=1e-4: {t_mlp:.2f} s")
    print(f"  CNN one epoch: {t_cnn_epoch:.2f} s")
    print("COUNTS OF WORK IN THE FULL RUN")
    print(f"  D2NN training runs: {n_train_runs} (3 DEV grid + 10 final + 3 C1), {EPOCHS} epochs x {n_tr} samples")
    print(f"  D2NN gradient calls: {grad_calls}")
    print(f"  D2NN forward calls: {fwd_calls}")
    print("ESTIMATES")
    for k, v in est.items():
        print(f"  {k}: {v:.0f} s ({v / 3600:.2f} h)")
    print(f"  total (approximate): {total:.0f} s ({total / 3600:.2f} h)")
    print(f"  rss_mb at end of dry run: {rss_mb()}")
    print("Dry run finished. No result files were written.")


# ---------------------------------------------------------------- selection and outputs
def select_beta(grid):
    """Sequential selection rule. grid is in ascending beta order; ties keep the smaller beta."""
    sel = None
    for row in grid:
        if sel is None or row["dev_acc"] > sel["dev_acc"]:
            sel = row
    return sel


def best_baseline(base):
    """Best baseline on DEV. Ties go to the baseline with more trainable parameters."""
    return sorted(base, key=lambda k: (-base[k]["dev_acc"], -base[k]["n_params"]))[0]


def finalize(data, d2nn_grid, sel_beta, base, best_name, pred_base, d2nn_runs, pred_d2nn,
             c1_runs, pred_c1, c2_runs, pred_c2, elapsed):
    """Statistics, verdicts and the two output files. Shared by --run and --part assemble.
    Expects base[name]["test_acc"] to be set already."""
    y_f = data["fresh"][1]
    K = len(y_f)
    # 6. Statistics. Per-seed Wilson intervals only; no pooled interval of the mean.
    best_pred = pred_base[best_name]
    for r in d2nn_runs:
        r["test_wilson_95"] = wilson(r["test_correct"], K)
        r["paired_vs_best_baseline"] = paired(pred_d2nn[r["seed"]], best_pred, y_f)
    for r in c1_runs + c2_runs:
        r["test_wilson_95"] = wilson(r["test_correct"], K)
    d2nn_accs = np.array([r["test_acc"] for r in d2nn_runs])
    mean_te = float(d2nn_accs.mean())
    min_te = float(d2nn_accs.min())
    for name in base:
        base[name]["test_correct"] = int(np.sum(pred_base[name] == y_f))
        base[name]["test_wilson_95"] = wilson(base[name]["test_correct"], K)
        base[name]["mean_diff_d2nn_minus_baseline"] = mean_te - base[name]["test_acc"]
        base[name]["mcnemar_seed0_vs_d2nn"] = paired(pred_d2nn[0], pred_base[name], y_f)
    c1_mean = float(np.mean([r["test_acc"] for r in c1_runs]))
    c2_mean = float(np.mean([r["test_acc"] for r in c2_runs]))
    mc0 = base[best_name]["mcnemar_seed0_vs_d2nn"]
    diff_best = mean_te - base[best_name]["test_acc"]

    verdicts = {
        "H1": "pass" if (mean_te >= 0.90 and min_te >= 0.88) else "fail",
        "H2": "pass" if (diff_best >= 0 and mc0["p_two_sided_exact"] < 0.05
                         and mc0["d2nn_only_correct"] > mc0["baseline_only_correct"]) else "fail",
        "C1": "pass" if c1_mean <= 0.15 else "fail",
        "C2": "pass" if c2_mean <= 0.20 else "fail",
    }
    summary = {
        "contract": "CONTRACT-007",
        "kind": "full run output (see docs/CONTRACTS/CONTRATO-007-test-fresco.md)",
        "data_counts": data["counts"],
        "fresh_rows_sha256": hashlib.sha256(np.asarray(data["fresh_rows"], dtype=np.int64).tobytes()).hexdigest(),
        "architecture": {"N": N, "dx_m": DX, "lambda_m": LAM, "distances_m": DIST,
                         "layers": 3, "phase_parameters": 3 * N * N, "detector_regions": 10},
        "d2nn": {"dev_grid": d2nn_grid, "selected_beta": sel_beta["beta"], "lr": LR_D2NN,
                 "epochs": EPOCHS, "batch": BATCH, "runs": d2nn_runs,
                 "test_mean": mean_te, "test_min": min_te,
                 "seed0_confusion_matrix_test_fresh": confusion(y_f, pred_d2nn[0])},
        "baselines": base,
        "best_baseline_on_dev": best_name,
        "controls": {"C1_permuted": {"seeds": C1_SEEDS, "mean": c1_mean, "runs": c1_runs},
                     "C2_untrained": {"seeds": C2_SEEDS, "mean": c2_mean, "runs": c2_runs}},
        "verdicts": verdicts,
        "decision_record": DECISION_RECORD,
        "elapsed_s": round(elapsed, 1),
    }
    full = dict(summary)
    full["predictions_test_fresh"] = {
        "d2nn_seed": {str(s): pred_d2nn[s] for s in D2NN_SEEDS},
        "baselines": dict(pred_base),
        "C1_seed": {str(s): pred_c1[s] for s in C1_SEEDS},
        "C2_seed": {str(s): pred_c2[s] for s in C2_SEEDS},
    }
    full["fresh_rows"] = data["fresh_rows"]
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1, default=_jsonable), encoding="utf-8")
    FULL_PATH.write_text(json.dumps(full, indent=1, default=_jsonable), encoding="utf-8")
    log(f"verdicts: {verdicts}")
    log(f"done in {elapsed:.0f} s")


# ---------------------------------------------------------------- full run
def run_full():
    global _WRITE_LOG
    _WRITE_LOG = True
    t0 = time.time()
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    data = prepare()
    A_tr, y_tr = data["tr"]
    A_dv, y_dv = data["dv"]
    A_f, y_f = data["fresh"]
    log(f"data: {data['counts']} rss={rss_mb()} MB")
    masks = build_masks()
    tested = set()

    def test_once(key):
        if key in tested:
            raise RuntimeError(f"TEST_FRESH already evaluated for {key}")
        tested.add(key)

    # 1. D2NN DEV grid, seed 0. No TEST_FRESH evaluation here.
    d2nn_grid = []
    for beta in BETA_GRID:                                 # ascending: ties keep the smaller beta
        model = make_d2nn(masks, beta, seed=0)
        train_d2nn(model, A_tr, y_tr, LR_D2NN, np.random.default_rng(0))
        a = float(np.mean(d2nn_predict(model, A_dv) == y_dv))
        d2nn_grid.append({"beta": beta, "lr": LR_D2NN, "dev_acc": a})
        log(f"D2NN DEV beta={beta}: {a:.4f}")
    sel_beta = select_beta(d2nn_grid)
    log(f"D2NN selected on DEV: beta={sel_beta['beta']} dev={sel_beta['dev_acc']:.4f}")

    # 2. Baselines: selection on DEV only. No TEST_FRESH evaluation here.
    base = {}
    grid_a, sel_a = select_logit(flat64(A_tr), y_tr, flat64(A_dv), y_dv)
    base["logit_64x64"] = {"selected": {"C": sel_a["C"]}, "dev_acc": sel_a["dev_acc"],
                           "dev_grid": grid_a, "n_params": sel_a["n_params"]}
    log(f"(a) logistic 64x64 selected C={sel_a['C']} dev={sel_a['dev_acc']:.4f}")
    grid_b, sel_b = select_logit(raw64(A_tr), y_tr, raw64(A_dv), y_dv)
    base["logit_8x8_raw"] = {"selected": {"C": sel_b["C"]}, "dev_acc": sel_b["dev_acc"],
                             "dev_grid": grid_b, "n_params": sel_b["n_params"]}
    log(f"(b) logistic 8x8 selected C={sel_b['C']} dev={sel_b['dev_acc']:.4f}")
    grid_c, sel_c = select_mlp(raw64(A_tr), y_tr, raw64(A_dv), y_dv)
    base["mlp_653"] = {"selected": {"alpha": sel_c["alpha"]}, "dev_acc": sel_c["dev_acc"],
                       "dev_grid": grid_c, "n_params": sel_c["n_params"]}
    log(f"(c) MLP selected alpha={sel_c['alpha']} dev={sel_c['dev_acc']:.4f}")
    torch.manual_seed(0)
    net = make_cnn()
    n_cnn = sum(p.numel() for p in net.parameters())
    dev_curve, cnn_best = train_cnn(net, to_tensor(A_tr), y_tr, np.random.default_rng(0),
                                    dev=(to_tensor(A_dv), y_dv))
    base["cnn"] = {"selected": {"epoch": cnn_best["epoch"]}, "dev_acc": cnn_best["dev_acc"],
                   "dev_curve": dev_curve, "n_params": int(n_cnn)}
    log(f"(d) CNN selected epoch={cnn_best['epoch']} dev={cnn_best['dev_acc']:.4f}")

    # Best baseline on DEV. Ties go to the baseline with more trainable parameters.
    best_name = best_baseline(base)
    log(f"best baseline on DEV: {best_name}")

    # 3. TEST_FRESH, once per baseline, after all selection.
    pred_base = {}
    test_once(("baseline", "logit_64x64"))
    pred_base["logit_64x64"] = predict_chunked(sel_a["model"].predict, flat64, A_f)
    test_once(("baseline", "logit_8x8_raw"))
    pred_base["logit_8x8_raw"] = predict_chunked(sel_b["model"].predict, raw64, A_f)
    test_once(("baseline", "mlp_653"))
    pred_base["mlp_653"] = predict_chunked(sel_c["model"].predict, raw64, A_f)
    test_once(("baseline", "cnn"))
    net_sel = make_cnn()
    net_sel.load_state_dict(cnn_best["state"])
    pred_base["cnn"] = cnn_predict(net_sel, to_tensor(A_f))
    for name in base:
        base[name]["test_acc"] = float(np.mean(pred_base[name] == y_f))
        log(f"TEST_FRESH baseline {name}: {base[name]['test_acc']:.4f}")
    del sel_a, sel_b, sel_c, net, net_sel

    # 4. D2NN final: 10 seeds, selected beta, TRAIN only; DEV and TEST_FRESH.
    d2nn_runs, pred_d2nn = [], {}
    for seed in D2NN_SEEDS:
        model = make_d2nn(masks, sel_beta["beta"], seed=seed)
        train_d2nn(model, A_tr, y_tr, LR_D2NN, np.random.default_rng(seed))
        dev_acc = float(np.mean(d2nn_predict(model, A_dv) == y_dv))
        test_once(("d2nn", seed))
        p = d2nn_predict(model, A_f)
        pred_d2nn[seed] = p
        d2nn_runs.append({"seed": seed, "dev_acc": dev_acc,
                          "test_acc": float(np.mean(p == y_f)), "test_correct": int(np.sum(p == y_f))})
        log(f"FINAL D2NN seed={seed}: DEV={dev_acc:.4f} TEST_FRESH={d2nn_runs[-1]['test_acc']:.4f}")

    # 5. Controls.
    c1_runs, pred_c1 = [], {}
    for seed in C1_SEEDS:
        perm = np.random.default_rng(100 + seed).permutation(y_tr)
        model = make_d2nn(masks, sel_beta["beta"], seed=seed)
        train_d2nn(model, A_tr, perm, LR_D2NN, np.random.default_rng(seed))
        test_once(("C1", seed))
        p = d2nn_predict(model, A_f)
        pred_c1[seed] = p
        c1_runs.append({"seed": seed, "test_acc": float(np.mean(p == y_f)), "test_correct": int(np.sum(p == y_f))})
        log(f"C1 permuted seed={seed}: TEST_FRESH={c1_runs[-1]['test_acc']:.4f}")
    c2_runs, pred_c2 = [], {}
    for seed in C2_SEEDS:
        model = make_d2nn(masks, sel_beta["beta"], seed=seed)          # untrained
        test_once(("C2", seed))
        p = d2nn_predict(model, A_f)
        pred_c2[seed] = p
        c2_runs.append({"seed": seed, "test_acc": float(np.mean(p == y_f)), "test_correct": int(np.sum(p == y_f))})
        log(f"C2 untrained seed={seed}: TEST_FRESH={c2_runs[-1]['test_acc']:.4f}")

    # 6. Statistics, verdicts and outputs (shared with --part assemble).
    finalize(data, d2nn_grid, sel_beta, base, best_name, pred_base, d2nn_runs, pred_d2nn,
             c1_runs, pred_c1, c2_runs, pred_c2, time.time() - t0)


# ---------------------------------------------------------------- parallel parts
# The same computations as run_full, split into independent processes. Each part writes one
# JSON file. Selection and verdicts are not repeated here: assemble applies select_beta,
# best_baseline and finalize, the code that --run uses. ':' is not allowed in Windows file
# names, so part "dev:0" is stored as dev_0.json.

def expected_parts():
    return ([f"dev:{k}" for k in range(len(BETA_GRID))]
            + [f"seed:{s}" for s in D2NN_SEEDS]
            + ["controls", "baselines"])


def parse_part(text):
    """--part value -> (kind, index, canonical part name). Raises ValueError if unknown."""
    if text in ("controls", "baselines", "assemble"):
        return text, None, text
    kind, sep, arg = text.partition(":")
    if sep and arg.isdigit():
        k = int(arg)
        if kind == "dev" and k in range(len(BETA_GRID)):
            return kind, k, f"dev:{k}"
        if kind == "seed" and k in D2NN_SEEDS:
            return kind, k, f"seed:{k}"
    raise ValueError(f"unknown --part {text!r}: use dev:K (K in {list(range(len(BETA_GRID)))}), "
                     f"seed:S (S in {D2NN_SEEDS}), controls, baselines or assemble")


def part_file(name):
    return PARTS_DIR / (name.replace(":", "_") + ".json")


def data_fingerprint(data):
    return hashlib.sha256(np.asarray(data["fresh_rows"], dtype=np.int64).tobytes()).hexdigest()


def load_part(name, data):
    """Whole part object. Refuses if the file is missing or was made from other data."""
    path = part_file(name)
    if not path.exists():
        raise SystemExit(f"missing part file for {name}: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("part") != name:
        raise SystemExit(f"{path} holds part {obj.get('part')!r}, expected {name!r}")
    if obj.get("data_counts") != data["counts"] or obj.get("fresh_rows_sha256") != data_fingerprint(data):
        raise SystemExit(f"{path} was made from different data; re-run that part")
    return obj


def load_dev_selection(data):
    """DEV grid and selected beta, read from the three dev parts with the sequential rule."""
    grid = []
    for k in range(len(BETA_GRID)):
        row = load_part(f"dev:{k}", data)["payload"]
        if row["beta"] != BETA_GRID[k] or row["lr"] != LR_D2NN:
            raise SystemExit(f"dev:{k} does not match BETA_GRID[{k}] and LR_D2NN")
        grid.append({"beta": row["beta"], "lr": row["lr"], "dev_acc": row["dev_acc"]})
    return grid, select_beta(grid)


def mark_test(tested, key):
    if key in tested:
        raise RuntimeError(f"TEST_FRESH already evaluated for {key}")
    tested.add(key)


def part_dev(data, k):
    A_tr, y_tr = data["tr"]
    A_dv, y_dv = data["dv"]
    beta = BETA_GRID[k]
    model = make_d2nn(build_masks(), beta, seed=0)
    train_d2nn(model, A_tr, y_tr, LR_D2NN, np.random.default_rng(0))
    a = float(np.mean(d2nn_predict(model, A_dv) == y_dv))
    log(f"D2NN DEV beta={beta}: {a:.4f}")
    return {"beta": beta, "lr": LR_D2NN, "dev_acc": a}


def part_seed(data, s):
    A_tr, y_tr = data["tr"]
    A_dv, y_dv = data["dv"]
    A_f, y_f = data["fresh"]
    _, sel = load_dev_selection(data)
    tested = set()
    model = make_d2nn(build_masks(), sel["beta"], seed=s)
    train_d2nn(model, A_tr, y_tr, LR_D2NN, np.random.default_rng(s))
    dev_acc = float(np.mean(d2nn_predict(model, A_dv) == y_dv))
    mark_test(tested, ("d2nn", s))
    p = d2nn_predict(model, A_f)
    log(f"FINAL D2NN seed={s}: DEV={dev_acc:.4f} TEST_FRESH={float(np.mean(p == y_f)):.4f}")
    return {"seed": s, "beta": sel["beta"], "dev_acc": dev_acc, "pred_test_fresh": p.tolist()}


def part_controls(data):
    A_tr, y_tr = data["tr"]
    A_f, y_f = data["fresh"]
    _, sel = load_dev_selection(data)
    masks = build_masks()
    tested = set()
    c1 = []
    for seed in C1_SEEDS:
        perm = np.random.default_rng(100 + seed).permutation(y_tr)
        model = make_d2nn(masks, sel["beta"], seed=seed)
        train_d2nn(model, A_tr, perm, LR_D2NN, np.random.default_rng(seed))
        mark_test(tested, ("C1", seed))
        p = d2nn_predict(model, A_f)
        log(f"C1 permuted seed={seed}: TEST_FRESH={float(np.mean(p == y_f)):.4f}")
        c1.append({"seed": seed, "pred_test_fresh": p.tolist()})
    c2 = []
    for seed in C2_SEEDS:
        model = make_d2nn(masks, sel["beta"], seed=seed)          # untrained
        mark_test(tested, ("C2", seed))
        p = d2nn_predict(model, A_f)
        log(f"C2 untrained seed={seed}: TEST_FRESH={float(np.mean(p == y_f)):.4f}")
        c2.append({"seed": seed, "pred_test_fresh": p.tolist()})
    return {"beta": sel["beta"], "c1": c1, "c2": c2}


def part_baselines(data):
    """Steps 2 and 3 of run_full: DEV selection for all four baselines, then TEST_FRESH once each."""
    A_tr, y_tr = data["tr"]
    A_dv, y_dv = data["dv"]
    A_f, y_f = data["fresh"]
    tested = set()
    base = {}
    grid_a, sel_a = select_logit(flat64(A_tr), y_tr, flat64(A_dv), y_dv)
    base["logit_64x64"] = {"selected": {"C": sel_a["C"]}, "dev_acc": sel_a["dev_acc"],
                           "dev_grid": grid_a, "n_params": sel_a["n_params"]}
    log(f"(a) logistic 64x64 selected C={sel_a['C']} dev={sel_a['dev_acc']:.4f}")
    grid_b, sel_b = select_logit(raw64(A_tr), y_tr, raw64(A_dv), y_dv)
    base["logit_8x8_raw"] = {"selected": {"C": sel_b["C"]}, "dev_acc": sel_b["dev_acc"],
                             "dev_grid": grid_b, "n_params": sel_b["n_params"]}
    log(f"(b) logistic 8x8 selected C={sel_b['C']} dev={sel_b['dev_acc']:.4f}")
    grid_c, sel_c = select_mlp(raw64(A_tr), y_tr, raw64(A_dv), y_dv)
    base["mlp_653"] = {"selected": {"alpha": sel_c["alpha"]}, "dev_acc": sel_c["dev_acc"],
                       "dev_grid": grid_c, "n_params": sel_c["n_params"]}
    log(f"(c) MLP selected alpha={sel_c['alpha']} dev={sel_c['dev_acc']:.4f}")
    torch.manual_seed(0)
    net = make_cnn()
    n_cnn = sum(p.numel() for p in net.parameters())
    dev_curve, cnn_best = train_cnn(net, to_tensor(A_tr), y_tr, np.random.default_rng(0),
                                    dev=(to_tensor(A_dv), y_dv))
    base["cnn"] = {"selected": {"epoch": cnn_best["epoch"]}, "dev_acc": cnn_best["dev_acc"],
                   "dev_curve": dev_curve, "n_params": int(n_cnn)}
    log(f"(d) CNN selected epoch={cnn_best['epoch']} dev={cnn_best['dev_acc']:.4f}")

    # TEST_FRESH, once per baseline, after all selection.
    pred_base = {}
    mark_test(tested, ("baseline", "logit_64x64"))
    pred_base["logit_64x64"] = predict_chunked(sel_a["model"].predict, flat64, A_f)
    mark_test(tested, ("baseline", "logit_8x8_raw"))
    pred_base["logit_8x8_raw"] = predict_chunked(sel_b["model"].predict, raw64, A_f)
    mark_test(tested, ("baseline", "mlp_653"))
    pred_base["mlp_653"] = predict_chunked(sel_c["model"].predict, raw64, A_f)
    mark_test(tested, ("baseline", "cnn"))
    net_sel = make_cnn()
    net_sel.load_state_dict(cnn_best["state"])
    pred_base["cnn"] = cnn_predict(net_sel, to_tensor(A_f))
    for name in base:
        log(f"TEST_FRESH baseline {name}: {float(np.mean(pred_base[name] == y_f)):.4f}")
    return {"base": base, "pred_test_fresh": {name: pred_base[name].tolist() for name in base}}


def run_part(text):
    kind, arg, name = parse_part(text)
    if kind == "assemble":
        assemble_parts()
        return
    path = part_file(name)
    if path.exists():
        raise SystemExit(f"refused: {path} already exists. TEST_FRESH is evaluated once per "
                         "configuration and seed; delete that file by hand only to re-run this part.")
    t_start = time.time()
    data = prepare()
    log(f"part {name}: data {data['counts']} rss={rss_mb()} MB")
    if kind == "dev":
        payload = part_dev(data, arg)
    elif kind == "seed":
        payload = part_seed(data, arg)
    elif kind == "controls":
        payload = part_controls(data)
    else:
        payload = part_baselines(data)
    obj = {"part": name, "contract": "CONTRACT-007", "data_counts": data["counts"],
           "fresh_rows_sha256": data_fingerprint(data), "started_epoch": t_start,
           "finished_epoch": time.time(), "payload": payload}
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_jsonable), encoding="utf-8")
    os.replace(tmp, path)
    log(f"part {name} written: {path} ({time.time() - t_start:.0f} s, rss={rss_mb()} MB)")


def assemble_parts():
    """Reads every expected part, applies the sequential selection and verdict code, and writes
    the same two outputs as --run. Refuses if any expected part file is missing."""
    global _WRITE_LOG
    missing = [n for n in expected_parts() if not part_file(n).exists()]
    if missing:
        raise SystemExit("assemble refused, part files missing: " + ", ".join(missing))
    _WRITE_LOG = True
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    data = prepare()
    LOG_PATH.write_text("", encoding="utf-8")
    log(f"assemble: data {data['counts']} rss={rss_mb()} MB")
    y_f = data["fresh"][1]
    objs = {n: load_part(n, data) for n in expected_parts()}

    # DEV selection of beta (same rule as --run)
    d2nn_grid, sel_beta = load_dev_selection(data)
    for row in d2nn_grid:
        log(f"D2NN DEV beta={row['beta']}: {row['dev_acc']:.4f}")
    log(f"D2NN selected on DEV: beta={sel_beta['beta']} dev={sel_beta['dev_acc']:.4f}")

    # Baselines: DEV selection was made in the baselines part; TEST_FRESH predictions are there.
    bp = objs["baselines"]["payload"]
    base = bp["base"]
    pred_base = {name: np.asarray(v, dtype=int) for name, v in bp["pred_test_fresh"].items()}
    if list(pred_base) != list(base):
        raise SystemExit("baselines part: prediction keys do not match the baseline list")
    for name in base:
        base[name]["test_acc"] = float(np.mean(pred_base[name] == y_f))
        log(f"TEST_FRESH baseline {name}: {base[name]['test_acc']:.4f}")
    best_name = best_baseline(base)
    log(f"best baseline on DEV: {best_name}")

    # D2NN final runs
    d2nn_runs, pred_d2nn = [], {}
    for seed in D2NN_SEEDS:
        sp = objs[f"seed:{seed}"]["payload"]
        if sp["seed"] != seed or sp["beta"] != sel_beta["beta"]:
            raise SystemExit(f"seed:{seed} was trained with beta={sp['beta']}, but DEV selects "
                             f"beta={sel_beta['beta']}; re-run seed:{seed}")
        p = np.asarray(sp["pred_test_fresh"], dtype=int)
        pred_d2nn[seed] = p
        d2nn_runs.append({"seed": seed, "dev_acc": sp["dev_acc"],
                          "test_acc": float(np.mean(p == y_f)), "test_correct": int(np.sum(p == y_f))})
        log(f"FINAL D2NN seed={seed}: DEV={sp['dev_acc']:.4f} TEST_FRESH={d2nn_runs[-1]['test_acc']:.4f}")

    # Controls
    cp = objs["controls"]["payload"]
    if cp["beta"] != sel_beta["beta"]:
        raise SystemExit(f"controls was run with beta={cp['beta']}, but DEV selects beta={sel_beta['beta']}")
    if [r["seed"] for r in cp["c1"]] != C1_SEEDS or [r["seed"] for r in cp["c2"]] != C2_SEEDS:
        raise SystemExit("controls part: seeds do not match C1_SEEDS and C2_SEEDS")
    c1_runs, pred_c1 = [], {}
    for r in cp["c1"]:
        p = np.asarray(r["pred_test_fresh"], dtype=int)
        pred_c1[r["seed"]] = p
        c1_runs.append({"seed": r["seed"], "test_acc": float(np.mean(p == y_f)),
                        "test_correct": int(np.sum(p == y_f))})
        log(f"C1 permuted seed={r['seed']}: TEST_FRESH={c1_runs[-1]['test_acc']:.4f}")
    c2_runs, pred_c2 = [], {}
    for r in cp["c2"]:
        p = np.asarray(r["pred_test_fresh"], dtype=int)
        pred_c2[r["seed"]] = p
        c2_runs.append({"seed": r["seed"], "test_acc": float(np.mean(p == y_f)),
                        "test_correct": int(np.sum(p == y_f))})
        log(f"C2 untrained seed={r['seed']}: TEST_FRESH={c2_runs[-1]['test_acc']:.4f}")

    # Wall-clock span of the parallel run (earliest part start to latest part end).
    elapsed = (max(o["finished_epoch"] for o in objs.values())
               - min(o["started_epoch"] for o in objs.values()))
    finalize(data, d2nn_grid, sel_beta, base, best_name, pred_base, d2nn_runs, pred_d2nn,
             c1_runs, pred_c1, c2_runs, pred_c2, elapsed)


def main():
    ap = argparse.ArgumentParser(description="CONTRACT-007 fresh-test benchmark")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--dry-run", action="store_true", help="counts and timing estimates; writes no results")
    g.add_argument("--run", action="store_true", help="full protocol (hours, CPU)")
    ap.add_argument("--part", metavar="PART", default=None,
                    help="one part of the parallel run: dev:K, seed:S, controls, baselines or assemble")
    args = ap.parse_args()
    if args.part is not None:
        if args.dry_run or args.run:
            ap.error("--part cannot be combined with --dry-run or --run")
        try:
            parse_part(args.part)
        except ValueError as e:
            ap.error(str(e))
        run_part(args.part)
    elif args.dry_run:
        dry_run()
    elif args.run:
        run_full()
    else:
        ap.error("one of the arguments --dry-run --run is required")


if __name__ == "__main__":
    main()
