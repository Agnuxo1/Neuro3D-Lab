"""CONTRATO-008: D2NN (CONTRATO-003 architecture) on MNIST and Fashion-MNIST, batched torch float64.

Implements docs/CONTRACTS/CONTRATO-008-mnist-fashion.md. This script runs the D2NN stages:
  select  DEV-only grid search (seed 0, first 10,000 TRAIN images, 5 epochs)
  final   full TRAIN, 20 epochs; TEST evaluated once per seed (seeds 0, 1, 2)
  c1      control: permuted TRAIN labels, DEV only (seed 0)
The digital baselines are in experiments/run_mnist_baselines.py.

Usage:
  python -B experiments/run_d2nn_mnist_torch.py --dataset mnist --stage select --seed 0
  python -B experiments/run_d2nn_mnist_torch.py --dataset mnist --stage final --seed 0
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from optlab.stats import wilson_interval  # noqa: E402
from run_d2nn_digits import DX as RDD_DX, N as RDD_N, build_masks  # noqa: E402

# ---- fixed by CONTRATO-008 ----
DATASETS = ["mnist", "fashion"]
DATA_ROOT = Path("D:/PROJECTS/.cognition/neuro3d-lab/datasets")
RESULTS_LOCAL = Path("D:/PROJECTS/.cognition/neuro3d-lab/results")
RESULTS_REPO = ROOT / "experiments" / "results"
SUMMARY_PATH = RESULTS_REPO / "mnist_fashion_summary.json"

TRAIN_IMG = "train-images-idx3-ubyte.gz"
TRAIN_LBL = "train-labels-idx1-ubyte.gz"
TEST_IMG = "t10k-images-idx3-ubyte.gz"
TEST_LBL = "t10k-labels-idx1-ubyte.gz"
SHA256 = {
    "mnist": {
        "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
    },
    "fashion": {
        "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
    },
}
N_OFFICIAL_TRAIN = 60_000
N_TRAIN = 50_000          # TRAIN = official training images 0..49,999
N_TEST = 10_000
SELECT_N = 10_000         # selection subset = first 10,000 of TRAIN
PAD = 32
N = 128
DX = 0.4e-3
LAM = 0.75e-3
DIST = [18e-3] * 4        # four propagations, L = 3 phase layers
L_LAYERS = len(DIST) - 1
N_DET = 10
BETAS = [20.0, 50.0, 100.0]
LRS = [0.02, 0.05]
SELECT_EPOCHS = 5
FINAL_EPOCHS = 20
BATCH = 32
EVAL_BATCH = 200
FINAL_SEEDS = [0, 1, 2]
SHUFFLE_OFFSET = 1000     # shuffle generator = default_rng(seed + 1000)
C1_OFFSET = 100           # label permutation = default_rng(100 + seed)
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
BASE_LR = 1e-3
BASE_EPOCHS = 20
BASELINES = ["logreg", "mlp12", "mlp128", "cnn"]
C1_TARGET = 0.10
C1_TOL = 0.03

# The detector geometry comes from run_d2nn_digits.py; it must match CONTRATO-008.
if (RDD_N, RDD_DX) != (N, DX):
    raise SystemExit("run_d2nn_digits.py geometry differs from CONTRATO-008")


# ---------------------------------------------------------------- utilities
def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- data
def _read_verified(dataset: str, name: str) -> bytes:
    raw = (DATA_ROOT / dataset / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = SHA256[dataset][name]
    if digest != expected:
        raise SystemExit(f"SHA-256 mismatch for {dataset}/{name}: {digest} != {expected}")
    return gzip.decompress(raw)


def _read_images(dataset: str, name: str, expected_n: int) -> np.ndarray:
    raw = _read_verified(dataset, name)
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051 or (rows, cols) != (28, 28) or n != expected_n:
        raise SystemExit(f"unexpected IDX image header in {dataset}/{name}")
    arr = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, rows, cols).copy()
    return arr


def _read_labels(dataset: str, name: str, expected_n: int) -> np.ndarray:
    raw = _read_verified(dataset, name)
    magic, n = struct.unpack(">II", raw[:8])
    if magic != 2049 or n != expected_n:
        raise SystemExit(f"unexpected IDX label header in {dataset}/{name}")
    arr = np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)
    if arr.size != n:
        raise SystemExit(f"label count mismatch in {dataset}/{name}")
    return arr


def load_split(dataset: str) -> dict:
    """TRAIN (images 0..49,999) and DEV (50,000..59,999). Never opens the TEST files."""
    x = _read_images(dataset, TRAIN_IMG, N_OFFICIAL_TRAIN)
    y = _read_labels(dataset, TRAIN_LBL, N_OFFICIAL_TRAIN)
    return {"x_train": x[:N_TRAIN], "y_train": y[:N_TRAIN],
            "x_dev": x[N_TRAIN:], "y_dev": y[N_TRAIN:]}


def load_test(dataset: str) -> dict:
    """Official TEST set. Call only after claim_test()."""
    return {"x": _read_images(dataset, TEST_IMG, N_TEST),
            "y": _read_labels(dataset, TEST_LBL, N_TEST)}


def encode64(x_u8: np.ndarray) -> torch.Tensor:
    """(B, 28, 28) uint8 -> (B, 64, 64) float64, unit L2 energy per image."""
    x = torch.from_numpy(np.ascontiguousarray(x_u8)).to(torch.float64) / 255.0
    x = F.interpolate(x.unsqueeze(1), size=(64, 64), mode="bilinear", align_corners=False).squeeze(1)
    x = x.clamp(0.0, 1.0)
    norm = torch.sqrt((x * x).sum(dim=(1, 2), keepdim=True))
    if bool((norm == 0).any()):
        raise ValueError("an image has zero energy after resize")
    return x / norm


def encode_field(x_u8: np.ndarray) -> torch.Tensor:
    """(B, 28, 28) uint8 -> (B, 128, 128) complex128, zero-padded by 32 px per side, zero phase."""
    u = F.pad(encode64(x_u8), (PAD, PAD, PAD, PAD))
    return torch.complex(u, torch.zeros_like(u))


# ---------------------------------------------------------------- optics
def transfer(d: float) -> torch.Tensor:
    """H = exp(i d kz) for propagating components, 0 for evanescent ones.
    Same definition as experiments/replicate_d2nn_pytorch.py (CONTRATO-009), copied here."""
    f = torch.fft.fftfreq(N, d=DX, dtype=torch.float64)
    fx, fy = torch.meshgrid(f, f, indexing="xy")
    kx = 2 * math.pi * fx
    ky = 2 * math.pi * fy
    k_wave = 2 * math.pi / LAM
    arg = k_wave**2 - kx**2 - ky**2
    prop = arg > 0
    kz = torch.sqrt(torch.where(prop, arg, torch.zeros_like(arg)))
    zero = torch.zeros((N, N), dtype=torch.complex128)
    return torch.where(prop, torch.polar(torch.ones_like(kz), d * kz), zero)


def setup_optics():
    masks_np = build_masks()                       # (10, N, N), from run_d2nn_digits.py
    if masks_np.shape != (N_DET, N, N) or masks_np.sum(axis=0).max() > 1.0:
        raise SystemExit("unexpected detector masks")
    masks = torch.from_numpy(np.asarray(masks_np, dtype=np.float64))
    H = [transfer(d) for d in DIST]
    return masks, H


def d2nn_scores(U: torch.Tensor, phases: torch.Tensor, masks: torch.Tensor, H, beta: float) -> torch.Tensor:
    """Forward pass for a batch: P, phase, P, phase, P, phase, P, then detector scores."""
    a = U
    for k in range(L_LAYERS):
        a = torch.fft.ifft2(H[k] * torch.fft.fft2(a)) * torch.polar(torch.ones_like(phases[k]), phases[k])
    f = torch.fft.ifft2(H[L_LAYERS] * torch.fft.fft2(a))
    intensity = f.real**2 + f.imag**2              # |f|^2 without the non-smooth sqrt
    return beta * torch.einsum("kij,bij->bk", masks, intensity)


@torch.no_grad()
def predict_d2nn(x_u8: np.ndarray, phases: torch.Tensor, masks, H, beta: float) -> np.ndarray:
    out = np.empty(len(x_u8), dtype=np.int64)
    for s in range(0, len(x_u8), EVAL_BATCH):
        sc = d2nn_scores(encode_field(x_u8[s:s + EVAL_BATCH]), phases, masks, H, beta)
        out[s:s + EVAL_BATCH] = sc.argmax(dim=1).numpy()
    return out


def train_d2nn(x_tr, y_tr, x_dev, y_dev, seed, beta, lr, epochs, masks, H, log):
    phases = torch.tensor(np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=(L_LAYERS, N, N)),
                          dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([phases], lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS)
    rng = np.random.default_rng(seed + SHUFFLE_OFFSET)
    history = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(x_tr))
        loss_sum = 0.0
        for s in range(0, len(order), BATCH):
            bi = order[s:s + BATCH]
            scores = d2nn_scores(encode_field(x_tr[bi]), phases, masks, H, beta)
            loss = F.cross_entropy(scores, torch.from_numpy(y_tr[bi]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(bi)
        dev_acc = float(np.mean(predict_d2nn(x_dev, phases.detach(), masks, H, beta) == y_dev))
        train_loss = loss_sum / len(order)
        history.append({"epoch": ep, "train_loss": train_loss, "dev_acc": dev_acc,
                        "seconds": round(time.time() - t0, 1)})
        log(f"beta={beta} lr={lr} seed={seed} epoch {ep}/{epochs}: "
            f"train_loss={train_loss:.4f} DEV={dev_acc:.4f}")
    return phases.detach(), history


# ---------------------------------------------------------------- TEST gate
def test_marker(dataset: str, model: str, seed: int) -> Path:
    return RESULTS_LOCAL / f"test_done_{dataset}_{model}_seed{seed}.marker"


def precheck_test(dataset: str, seed: int, models: list) -> None:
    """Refuse to start (before any training) if a TEST marker already exists for any model."""
    clash = [m for m in models if test_marker(dataset, m, seed).exists()]
    if clash:
        raise SystemExit(f"REFUSED: TEST already claimed for {clash} ({dataset}, seed {seed}); "
                         f"markers in {RESULTS_LOCAL}. Not starting.")


def claim_test(dataset: str, seed: int, models: list, log) -> None:
    """One marker file per dataset, model and seed, created exclusively (O_CREAT | O_EXCL).
    Atomic: two processes cannot both claim the same model. FileExistsError -> refuse TEST."""
    RESULTS_LOCAL.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for m in models:
        path = test_marker(dataset, m, seed)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"REFUSED: TEST already claimed for {m} ({dataset}, seed {seed}); marker {path}.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"contract": "CONTRATO-008", "dataset": dataset, "model": m,
                                 "seed": seed, "claimed": stamp}) + "\n")
        log(f"TEST claimed for {m} (marker {path.name})")


def read_selection(dataset: str) -> dict:
    path = RESULTS_LOCAL / f"{dataset}_select.json"
    if not path.exists():
        raise SystemExit(f"no selection for {dataset}: run --stage select first ({path})")
    return json.loads(path.read_text(encoding="utf-8"))["selected"]


# ---------------------------------------------------------------- stages
def stage_select(dataset: str, log) -> None:
    out = RESULTS_LOCAL / f"{dataset}_select.json"
    if out.exists():
        raise SystemExit(f"selection already frozen: {out}")
    data = load_split(dataset)
    x_sel, y_sel = data["x_train"][:SELECT_N], data["y_train"][:SELECT_N]
    masks, H = setup_optics()
    grid = []
    for beta in BETAS:
        for lr in LRS:
            _, hist = train_d2nn(x_sel, y_sel, data["x_dev"], data["y_dev"], 0, beta, lr,
                                 SELECT_EPOCHS, masks, H, log)
            grid.append({"beta": beta, "lr": lr, "dev_acc": hist[-1]["dev_acc"], "history": hist})
            log(f"SELECT beta={beta} lr={lr}: DEV={hist[-1]['dev_acc']:.4f}")
    best = sorted(grid, key=lambda r: (-r["dev_acc"], r["beta"], r["lr"]))[0]
    edge = {"beta": best["beta"] in (min(BETAS), max(BETAS)),
            "lr": best["lr"] in (min(LRS), max(LRS))}
    edge["any"] = edge["beta"] or edge["lr"]
    selected = {"beta": best["beta"], "lr": best["lr"], "dev_acc": best["dev_acc"], "edge": edge}
    write_json(out, {"contract": "CONTRATO-008", "dataset": dataset, "stage": "select", "seed": 0,
                     "subset": {"train_first_n": SELECT_N, "epochs": SELECT_EPOCHS},
                     "grid": grid, "selected": selected})
    log(f"SELECTED {selected}" + (" (on a grid edge: report as a limitation, grid not extended)" if edge["any"] else ""))
    write_summary(log)


def stage_final(dataset: str, seed: int, log) -> None:
    phases_name = f"{dataset}_final_seed{seed}_phases.npy"
    phases_path = RESULTS_LOCAL / phases_name
    precheck_test(dataset, seed, ["d2nn"])
    if phases_path.exists():
        raise SystemExit(f"REFUSED: phases file already exists, never overwritten: {phases_path}")
    sel = read_selection(dataset)
    beta, lr = sel["beta"], sel["lr"]
    data = load_split(dataset)
    masks, H = setup_optics()
    phases, hist = train_d2nn(data["x_train"], data["y_train"], data["x_dev"], data["y_dev"],
                              seed, beta, lr, FINAL_EPOCHS, masks, H, log)
    with phases_path.open("xb") as fh:      # 'x': never overwrite saved phases
        np.save(fh, phases.numpy())
    claim_test(dataset, seed, ["d2nn"], log)
    test = load_test(dataset)
    pred = predict_d2nn(test["x"], phases, masks, H, beta)
    correct = pred == test["y"]
    k = int(correct.sum())
    result = {
        "contract": "CONTRATO-008", "dataset": dataset, "model": "d2nn", "seed": seed,
        "beta": beta, "lr": lr, "epochs": FINAL_EPOCHS, "batch": BATCH,
        "params": int(phases.numel()),
        "dev_acc_final": hist[-1]["dev_acc"], "dev_history": hist,
        "test_acc": k / N_TEST, "test_wilson95": list(wilson_interval(k, N_TEST)),
        "test_pred": pred.tolist(), "test_correct": correct.tolist(),
        "phases_file": phases_name,
    }
    write_json(RESULTS_LOCAL / f"{dataset}_final_seed{seed}.json", result)
    log(f"TEST d2nn seed {seed}: {k / N_TEST:.4f}")
    write_summary(log)


def stage_c1(dataset: str, log) -> None:
    sel = read_selection(dataset)
    beta, lr = sel["beta"], sel["lr"]
    data = load_split(dataset)
    y_perm = np.random.default_rng(C1_OFFSET + 0).permutation(data["y_train"])
    masks, H = setup_optics()
    _, hist = train_d2nn(data["x_train"], y_perm, data["x_dev"], data["y_dev"], 0, beta, lr,
                         FINAL_EPOCHS, masks, H, log)
    dev = hist[-1]["dev_acc"]
    result = {
        "contract": "CONTRATO-008", "dataset": dataset, "control": "C1", "seed": 0,
        "beta": beta, "lr": lr, "epochs": FINAL_EPOCHS,
        "dev_acc_final": dev, "dev_history": hist,
        "expectation": {"target": C1_TARGET, "tolerance": C1_TOL,
                        "met": bool(abs(dev - C1_TARGET) <= C1_TOL),
                        "note": "pre-registered expectation, not a verdict"},
    }
    write_json(RESULTS_LOCAL / f"{dataset}_c1_seed0.json", result)
    log(f"C1 DEV={dev:.4f} (expectation within {C1_TOL} of {C1_TARGET}: {result['expectation']['met']})")
    write_summary(log)


# ---------------------------------------------------------------- summary
def _load_run(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _mean_sd(vals: list) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "sd_sample": None}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd_sample": float(a.std(ddof=1)) if a.size > 1 else None}


def holm_adjust(pvals: list) -> list:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def _mcnemar_block(d2nn0, base0) -> dict:
    if d2nn0 is None or base0 is None:
        return {"status": "pending"}
    d_corr = np.asarray(d2nn0["test_correct"], dtype=bool)
    raw, rows = [], {}
    for name in BASELINES:
        if name not in base0["models"]:
            return {"status": "pending", "missing": name}
        b_corr = np.asarray(base0["models"][name]["test_correct"], dtype=bool)
        b = int(np.sum(d_corr & ~b_corr))
        c = int(np.sum(~d_corr & b_corr))
        p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c > 0 else 1.0
        raw.append(p)
        rows[name] = {"b_d2nn_only_correct": b, "c_baseline_only_correct": c, "p_raw": p,
                      "acc_d2nn_seed0": float(d_corr.mean()), "acc_baseline_seed0": float(b_corr.mean())}
    for (name, row), p_adj in zip(rows.items(), holm_adjust(raw)):
        row["p_holm"] = p_adj
        beats = p_adj < 0.05 and row["acc_d2nn_seed0"] > row["acc_baseline_seed0"]
        row["verdict"] = "D2NN beats baseline" if beats else "D2NN does not beat baseline"
    return {"status": "complete", "comparisons": rows}


def _summarize_dataset(ds: str) -> dict:
    sel = _load_run(RESULTS_LOCAL / f"{ds}_select.json")
    finals = {s: _load_run(RESULTS_LOCAL / f"{ds}_final_seed{s}.json") for s in FINAL_SEEDS}
    finals = {s: r for s, r in finals.items() if r is not None}
    bases = {s: _load_run(RESULTS_LOCAL / f"{ds}_baselines_seed{s}.json") for s in FINAL_SEEDS}
    bases = {s: r for s, r in bases.items() if r is not None}
    c1 = _load_run(RESULTS_LOCAL / f"{ds}_c1_seed0.json")

    d2nn_block = {
        "selected": None if sel is None else sel["selected"],
        "params": 3 * N * N,
        "seeds": [{"seed": s, "dev_acc_final": r["dev_acc_final"], "test_acc": r["test_acc"],
                   "test_wilson95": r["test_wilson95"]} for s, r in sorted(finals.items())],
        "test_acc_across_seeds": _mean_sd([r["test_acc"] for r in finals.values()]),
    }
    base_block = {}
    for name in BASELINES:
        seeds = [{"seed": s, "dev_acc_final": r["models"][name]["dev_acc_final"],
                  "test_acc": r["models"][name]["test_acc"],
                  "test_wilson95": r["models"][name]["test_wilson95"]}
                 for s, r in sorted(bases.items()) if name in r["models"]]
        params = next((r["models"][name]["params"] for r in bases.values() if name in r["models"]), None)
        base_block[name] = {"params": params, "seeds": seeds,
                            "test_acc_across_seeds": _mean_sd([x["test_acc"] for x in seeds])}
    return {
        "d2nn": d2nn_block,
        "baselines": base_block,
        "mcnemar_seed0_TEST": _mcnemar_block(finals.get(0), bases.get(0)),
        "c1": None if c1 is None else {"dev_acc_final": c1["dev_acc_final"], "expectation": c1["expectation"]},
    }


def write_summary(log=None) -> None:
    summary = {
        "contract": "CONTRATO-008",
        "caveat": ("All D2NN numbers use the periodic FFT window (51.2 mm at dx = 0.4 mm; CONTRATO-006c). "
                   "The periodic-window caveat (CONTRATO-010, pending) applies to every D2NN number here."),
        "intervals": "Wilson 95 % per seed and model (n = 10,000). No binomial interval for seed means (CONTRATO-003 note 4).",
        "datasets": {ds: _summarize_dataset(ds) for ds in DATASETS},
    }
    write_json(SUMMARY_PATH, summary)
    if log is not None:
        log(f"summary written: {SUMMARY_PATH}")


# ---------------------------------------------------------------- entry point
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--stage", required=True, choices=["select", "final", "baselines", "c1"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.stage == "baselines":
        ap.error("the baselines are run by experiments/run_mnist_baselines.py")
    if args.stage in ("select", "c1") and args.seed != 0:
        ap.error(f"--stage {args.stage} is registered for seed 0 only")
    if args.stage == "final" and args.seed not in FINAL_SEEDS:
        ap.error(f"--seed must be one of {FINAL_SEEDS}")

    torch.set_default_dtype(torch.float64)
    log = make_logger(RESULTS_LOCAL / f"{args.dataset}_{args.stage}_seed{args.seed}.log")
    log(f"start: dataset={args.dataset} stage={args.stage} seed={args.seed}")
    t0 = time.time()
    if args.stage == "select":
        stage_select(args.dataset, log)
    elif args.stage == "final":
        stage_final(args.dataset, args.seed, log)
    elif args.stage == "c1":
        stage_c1(args.dataset, log)
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
