"""CONTRATO-008: digital baselines on MNIST and Fashion-MNIST, same splits and budget as the D2NN.

Implements docs/CONTRACTS/CONTRATO-008-mnist-fashion.md, stage 'baselines' only.
Models: logreg (4096 -> 10), mlp12 (parameter-matched to the D2NN), mlp128 (stronger digital
reference), cnn (small CNN). Adam lr 1e-3, betas (0.9, 0.999), eps 1e-8, batch 32, 20 epochs,
final-epoch model, float64 on CPU. Inputs are the 64 x 64 normalised images, with no padding.

Usage:
  python -B experiments/run_mnist_baselines.py --dataset mnist --stage baselines --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_d2nn_mnist_torch import (  # noqa: E402
    ADAM_BETAS,
    ADAM_EPS,
    BASE_EPOCHS,
    BASE_LR,
    BASELINES,
    BATCH,
    DATASETS,
    EVAL_BATCH,
    FINAL_SEEDS,
    N_TEST,
    RESULTS_LOCAL,
    SHUFFLE_OFFSET,
    claim_test,
    encode64,
    load_split,
    load_test,
    precheck_test,
    make_logger,
    wilson_interval,
    write_json,
    write_summary,
)


def build_model(name: str) -> nn.Module:
    if name == "logreg":
        net = nn.Sequential(nn.Flatten(), nn.Linear(4096, 10))
    elif name == "mlp12":
        net = nn.Sequential(nn.Flatten(), nn.Linear(4096, 12), nn.ReLU(), nn.Linear(12, 10))
    elif name == "mlp128":
        net = nn.Sequential(nn.Flatten(), nn.Linear(4096, 128), nn.ReLU(), nn.Linear(128, 10))
    elif name == "cnn":
        net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 16 * 16, 10),
        )
    else:
        raise ValueError(f"unknown baseline: {name}")
    return net.double()


@torch.no_grad()
def predict(model: nn.Module, x_u8: np.ndarray) -> np.ndarray:
    model.eval()
    out = np.empty(len(x_u8), dtype=np.int64)
    for s in range(0, len(x_u8), EVAL_BATCH):
        X = encode64(x_u8[s:s + EVAL_BATCH]).unsqueeze(1)
        out[s:s + EVAL_BATCH] = model(X).argmax(dim=1).numpy()
    return out


def train_model(name: str, x_tr, y_tr, x_dev, y_dev, seed: int, log):
    torch.manual_seed(seed)
    model = build_model(name)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR, betas=ADAM_BETAS, eps=ADAM_EPS)
    rng = np.random.default_rng(seed + SHUFFLE_OFFSET)
    history = []
    for ep in range(1, BASE_EPOCHS + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(x_tr))
        loss_sum = 0.0
        for s in range(0, len(order), BATCH):
            bi = order[s:s + BATCH]
            X = encode64(x_tr[bi]).unsqueeze(1)
            loss = F.cross_entropy(model(X), torch.from_numpy(y_tr[bi]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(bi)
        dev_acc = float(np.mean(predict(model, x_dev) == y_dev))
        train_loss = loss_sum / len(order)
        history.append({"epoch": ep, "train_loss": train_loss, "dev_acc": dev_acc,
                        "seconds": round(time.time() - t0, 1)})
        log(f"{name} seed={seed} epoch {ep}/{BASE_EPOCHS}: train_loss={train_loss:.4f} DEV={dev_acc:.4f}")
    return model, n_params, history


def stage_baselines(dataset: str, seed: int, log) -> None:
    precheck_test(dataset, seed, BASELINES)    # refuse before any training if a marker exists
    data = load_split(dataset)
    trained = {}
    for name in BASELINES:
        trained[name] = train_model(name, data["x_train"], data["y_train"],
                                    data["x_dev"], data["y_dev"], seed, log)
    # One-use TEST claim (one exclusive marker file per model), made after training and before any TEST prediction.
    claim_test(dataset, seed, BASELINES, log)
    test = load_test(dataset)
    models_out = {}
    for name in BASELINES:
        model, n_params, hist = trained[name]
        pred = predict(model, test["x"])
        correct = pred == test["y"]
        k = int(correct.sum())
        models_out[name] = {
            "params": int(n_params),
            "dev_acc_final": hist[-1]["dev_acc"], "dev_history": hist,
            "test_acc": k / N_TEST, "test_wilson95": list(wilson_interval(k, N_TEST)),
            "test_pred": pred.tolist(), "test_correct": correct.tolist(),
        }
        log(f"TEST {name} seed {seed}: {k / N_TEST:.4f}")
    write_json(RESULTS_LOCAL / f"{dataset}_baselines_seed{seed}.json", {
        "contract": "CONTRATO-008", "dataset": dataset, "seed": seed,
        "epochs": BASE_EPOCHS, "lr": BASE_LR, "batch": BATCH, "models": models_out,
    })
    write_summary(log)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--stage", required=True, choices=["select", "final", "baselines", "c1"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.stage != "baselines":
        ap.error("this script runs --stage baselines only; the D2NN stages are in run_d2nn_mnist_torch.py")
    if args.seed not in FINAL_SEEDS:
        ap.error(f"--seed must be one of {FINAL_SEEDS}")

    torch.set_default_dtype(torch.float64)
    log = make_logger(RESULTS_LOCAL / f"{args.dataset}_baselines_seed{args.seed}.log")
    log(f"start: dataset={args.dataset} stage=baselines seed={args.seed}")
    t0 = time.time()
    stage_baselines(args.dataset, args.seed, log)
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
