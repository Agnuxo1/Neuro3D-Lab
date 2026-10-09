"""CONTRATO-003: red difractiva de fase (3 capas) en dígitos manuscritos.

Implementa el protocolo preregistrado en docs/CONTRACTS/CONTRATO-003-d2nn-digitos.md
(incluido su addendum). Los umbrales no se cambian aquí; los veredictos se calculan
sólo al final, con TEST usado una única vez.

Uso: python experiments/run_d2nn_digits.py
Salida: resumen en experiments/results/d2nn_digits_summary.json (versionado) y datos
completos (predicciones, matriz de confusión, historial DEV) en
D:/PROJECTS/.cognition/neuro3d-lab/results/d2nn_digits.json (no versionado).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optlab.d2nn import DiffractiveNetwork  # noqa: E402
from optlab.elements import detector_masks  # noqa: E402
from optlab.stats import wilson_interval  # noqa: E402

# ---- parámetros fijados en el contrato ----
SPLIT_SEED = 2026
N = 128
DX = 0.4e-3
LAM = 0.75e-3
DIST = [18e-3] * 4          # entrada -> capa1 -> capa2 -> capa3 -> detector
L_LAYERS = 3
BETAS = [20.0, 50.0, 100.0]
LRS = [0.02, 0.05]
EPOCHS = 30
BATCH = 32
SEEDS = [0, 1, 2, 3, 4]
C_GRID = [0.01, 0.1, 1.0, 10.0]
OUT_LOCAL = Path("D:/PROJECTS/.cognition/neuro3d-lab/results")
OUT_REPO = ROOT / "experiments" / "results"
LOG = OUT_LOCAL / "d2nn_digits.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_split():
    d = load_digits()
    img8 = d.images / 16.0
    y = d.target
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.4, stratify=y, random_state=SPLIT_SEED)
    dv, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=SPLIT_SEED)
    return img8, y, tr, dv, te


def encode64(img8_one):
    a = np.kron(img8_one, np.ones((8, 8)))          # 64x64 por repetición
    return a / np.linalg.norm(a)                     # energía unitaria por muestra


def encode_field(img8_one):
    a = encode64(img8_one)
    return np.pad(a, 32).astype(complex)             # centrado en 128x128


def build_masks():
    cols = [-40, -20, 0, 20, 40]                     # píxeles, +x a la derecha
    rows = [20, -20]                                 # +y arriba: fila 0 es la superior
    regions = []
    for k in range(10):
        cx = cols[k % 5] * DX
        cy = rows[k // 5] * DX
        regions.append({"cx_m": cx, "cy_m": cy, "w_m": 12 * DX, "h_m": 12 * DX})
    return detector_masks(N, DX, regions)


def make_model(masks, beta, seed):
    return DiffractiveNetwork(N, DX, LAM, DIST, masks, scale=beta, seed=seed)


def train(model, U, y, lr, rng, epochs=EPOCHS, batch=BATCH):
    """Adam por mini-lotes con el gradiente analítico de optlab."""
    m = np.zeros_like(model.phases)
    v = np.zeros_like(model.phases)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    idx = np.arange(len(U))
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            bi = idx[s:s + batch]
            g = np.zeros_like(model.phases)
            for i in bi:
                _, gi = model.loss_and_grad(U[i], int(y[i]))
                g += gi
            g /= len(bi)
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            model.phases -= lr * mh / (np.sqrt(vh) + eps)
    return model


def predictions(model, U):
    return np.array([model.predict(u) for u in U])


def acc(pred, y):
    return float(np.mean(pred == y))


def logistic_acc(Xtr, ytr, Xev, yev, C):
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(Xtr, ytr)
    return clf.predict(Xev)


def main():
    t0 = time.time()
    OUT_LOCAL.mkdir(parents=True, exist_ok=True)
    OUT_REPO.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8")
    img8, y_all, tr, dv, te = load_split()
    log(f"partición: TRAIN={len(tr)} DEV={len(dv)} TEST={len(te)} (semilla {SPLIT_SEED})")

    masks = build_masks()
    U = {name: [encode_field(img8[i]) for i in ids] for name, ids in
         (("tr", tr), ("dv", dv), ("te", te))}
    y = {"tr": y_all[tr], "dv": y_all[dv], "te": y_all[te]}

    # ---- 1. selección en DEV (seed 0, sólo TRAIN y DEV) ----
    dev_grid = []
    for beta in BETAS:
        for lr in LRS:
            model = make_model(masks, beta, seed=0)
            train(model, U["tr"], y["tr"], lr, np.random.default_rng(0))
            a = acc(predictions(model, U["dv"]), y["dv"])
            dev_grid.append({"beta": beta, "lr": lr, "dev_acc": a})
            log(f"DEV beta={beta} lr={lr}: acc={a:.4f}")
    best = sorted(dev_grid, key=lambda r: (-r["dev_acc"], r["beta"], r["lr"]))[0]
    log(f"configuración elegida en DEV: beta={best['beta']} lr={best['lr']} acc={best['dev_acc']:.4f}")

    # ---- 2. final: 5 seeds sobre TRAIN, DEV y TEST (TEST una sola vez) ----
    final = []
    te_preds_seed0 = None
    for seed in SEEDS:
        model = make_model(masks, best["beta"], seed=seed)
        train(model, U["tr"], y["tr"], best["lr"], np.random.default_rng(seed))
        pdv = predictions(model, U["dv"])
        pte = predictions(model, U["te"])
        if seed == 0:
            te_preds_seed0 = pte
        final.append({"seed": seed, "dev_acc": acc(pdv, y["dv"]), "test_acc": acc(pte, y["te"])})
        log(f"FINAL seed={seed}: DEV={final[-1]['dev_acc']:.4f} TEST={final[-1]['test_acc']:.4f}")
    test_accs = np.array([f["test_acc"] for f in final])

    # ---- 3. controles ----
    c1 = []
    for seed in SEEDS:
        perm = np.random.default_rng(100 + seed).permutation(y["tr"])
        model = make_model(masks, best["beta"], seed=seed)
        train(model, U["tr"], perm, best["lr"], np.random.default_rng(seed))
        c1.append(acc(predictions(model, U["te"]), y["te"]))
        log(f"C1 (etiquetas permutadas) seed={seed}: TEST={c1[-1]:.4f}")
    c2 = []
    for seed in SEEDS:
        model = make_model(masks, best["beta"], seed=seed)  # sin entrenar
        c2.append(acc(predictions(model, U["te"]), y["te"]))
    log(f"C2 (sin entrenar) TEST: {[round(a, 4) for a in c2]}")
    zero = make_model(masks, best["beta"], seed=0)
    zero.phases[:] = 0.0                                    # propagación libre por tramos
    c3 = acc(predictions(zero, U["te"]), y["te"])
    log(f"C3 (fases a cero) TEST={c3:.4f}")

    # ---- 4. bases de comparación (sólo se usan para H2) ----
    X64 = {k: np.array([encode64(img8[i]).ravel() for i in ids]) for k, ids in
           (("tr", tr), ("dv", dv), ("te", te))}
    X8 = {k: np.array([(img8[i] / np.linalg.norm(img8[i])).ravel() for i in ids]) for k, ids in
          (("tr", tr), ("dv", dv), ("te", te))}
    base = {}
    for name, XX in (("logistica_64x64", X64), ("logistica_8x8", X8)):
        dev_c = []
        for C in C_GRID:
            p = logistic_acc(XX["tr"], y["tr"], XX["dv"], y["dv"], C)
            dev_c.append((acc(p, y["dv"]), C))
        bestC = sorted(dev_c, key=lambda r: (-r[0], r[1]))[0][1]
        pte = logistic_acc(XX["tr"], y["tr"], XX["te"], y["te"], bestC)
        base[name] = {"C": bestC, "dev_grid": dev_c, "test_acc": acc(pte, y["te"]), "pred": pte}
        log(f"{name}: C={bestC} TEST={base[name]['test_acc']:.4f}")

    # ---- 5. H2: McNemar exacto sobre seed 0 frente a la logística 64x64 ----
    base_pred = base["logistica_64x64"]["pred"]
    corr_d = te_preds_seed0 == y["te"]
    corr_b = base_pred == y["te"]
    b = int(np.sum(corr_d & ~corr_b))     # D2NN acierta, base falla
    c = int(np.sum(~corr_d & corr_b))     # D2NN falla, base acierta
    p_mc = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c > 0 else 1.0
    diff = float(test_accs.mean() - base["logistica_64x64"]["test_acc"])

    # ---- 6. veredictos ----
    mean_te = float(test_accs.mean())
    min_te = float(test_accs.min())
    verdicts = {
        "H1": "cumple" if (mean_te >= 0.85 and min_te >= 0.80) else "no cumple",
        "H2": "cumple" if (diff >= 0 and p_mc < 0.05) else "no cumple",
        "C1": "cumple" if np.mean(c1) <= 0.15 else "no cumple",
        "C2": "cumple" if np.mean(c2) <= 0.20 else "no cumple",
        "C3": "cumple" if c3 <= 0.20 else "no cumple",
    }
    k = int(len(te))
    wil = {"d2nn_mean_seeds": wilson_interval(int(round(mean_te * k)), k)}
    results = {
        "contract": "CONTRATO-003", "split_seed": SPLIT_SEED,
        "sizes": {"train": len(tr), "dev": len(dv), "test": len(te)},
        "architecture": {"N": N, "dx_m": DX, "lambda_m": LAM, "distances_m": DIST, "layers": L_LAYERS},
        "dev_grid": dev_grid, "selected": best, "final": final,
        "test_mean": mean_te, "test_min": min_te, "test_wilson_mean_seeds": wil["d2nn_mean_seeds"],
        "controls": {"C1_permuted_mean": float(np.mean(c1)), "C1_per_seed": c1,
                     "C2_untrained_mean": float(np.mean(c2)), "C2_per_seed": c2,
                     "C3_zero_phase": c3},
        "baselines": {n: {"C": v["C"], "dev_grid": v["dev_grid"], "test_acc": v["test_acc"],
                          "test_wilson": wilson_interval(int(round(v["test_acc"] * k)), k)}
                      for n, v in base.items()},
        "mcnemar": {"b_d2nn_only_correct": b, "c_base_only_correct": c, "p_two_sided": p_mc},
        "diff_mean_minus_logistic64": diff,
        "verdicts": verdicts,
        "confusion_matrix_seed0": confusion_matrix(y["te"], te_preds_seed0).tolist(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_LOCAL / "d2nn_digits.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    summary = {k2: results[k2] for k2 in ("contract", "sizes", "architecture", "selected", "final",
                                          "test_mean", "test_min", "test_wilson_mean_seeds",
                                          "controls", "baselines", "mcnemar",
                                          "diff_mean_minus_logistic64", "verdicts", "elapsed_s")}
    summary["dev_grid"] = dev_grid
    (OUT_REPO / "d2nn_digits_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"VEREDICTOS: {verdicts}")
    log(f"TEST media seeds={mean_te:.4f} mín={min_te:.4f} | logística64={base['logistica_64x64']['test_acc']:.4f} "
        f"| McNemar b={b} c={c} p={p_mc:.4g}")
    log(f"fin en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
