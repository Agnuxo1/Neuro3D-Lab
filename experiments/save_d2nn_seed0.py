"""Reproducción post hoc del modelo D2NN de CONTRATO-003 (semilla 0), entrenado sólo con TRAIN.

Reproducción post hoc, no un veredicto nuevo: reentrena la semilla 0 con beta=100.0 y lr=0.02
(valores elegidos en DEV por run_d2nn_digits.py), comprueba que la precisión en DEV coincide
con la registrada en experiments/results/d2nn_digits_summary.json y guarda las fases junto con
su SHA-256. TEST no se codifica, no se carga para evaluación y no se evalúa en este script.

Reutiliza las funciones de experiments/run_d2nn_digits.py (importar no ejecuta main()).

Uso: python -B experiments/save_d2nn_seed0.py
Salida: experiments/results/d2nn_seed0_phases.npz y experiments/results/d2nn_seed0_checksum.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # evita escribir __pycache__ dentro del repositorio

EXPERIMENTS = Path(__file__).resolve().parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT / "src"))

import run_d2nn_digits as R  # noqa: E402  (importar no ejecuta main())

BETA = 100.0
LR = 0.02
SEED = 0
EXPECTED_DEV_ACC = 0.9554317548746518
DEV_TOL = 1e-12

OUT_DIR = EXPERIMENTS / "results"
NPZ_PATH = OUT_DIR / "d2nn_seed0_phases.npz"
CHECKSUM_PATH = OUT_DIR / "d2nn_seed0_checksum.json"


def main():
    # Mismo split que run_d2nn_digits.main(). El índice de TEST se descarta sin codificarlo.
    img8, y_all, tr, dv, _te_unused = R.load_split()
    masks = R.build_masks()

    U_tr = [R.encode_field(img8[i]) for i in tr]
    U_dv = [R.encode_field(img8[i]) for i in dv]
    y_tr = y_all[tr]
    y_dv = y_all[dv]

    # Mismo entrenamiento que la semilla 0 de main(): mismo generador de barajado.
    model = R.make_model(masks, BETA, seed=SEED)
    R.train(model, U_tr, y_tr, LR, np.random.default_rng(0))

    dev_acc = R.acc(R.predictions(model, U_dv), y_dv)
    print(f"DEV beta={BETA} lr={LR} seed={SEED}: acc={dev_acc!r}", flush=True)
    if abs(dev_acc - EXPECTED_DEV_ACC) > DEV_TOL:
        raise RuntimeError(
            f"La precisión en DEV ({dev_acc!r}) no coincide con {EXPECTED_DEV_ACC!r} "
            f"dentro de {DEV_TOL}. No se guarda nada."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        NPZ_PATH,
        phases=np.asarray(model.phases),
        beta=np.float64(BETA),
        lr=np.float64(LR),
        seed=np.int64(SEED),
        dev_acc=np.float64(dev_acc),
        distances_m=np.asarray(R.DIST, dtype=np.float64),
        pixel_m=np.float64(R.DX),
        wavelength_m=np.float64(R.LAM),
    )

    data = NPZ_PATH.read_bytes()
    checksum = {
        "file": NPZ_PATH.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "contract": "CONTRATO-003",
        "seed": SEED,
        "beta": BETA,
        "lr": LR,
        "dev_acc": dev_acc,
        "note": "Reproducción post hoc entrenada sólo con TRAIN; no es un veredicto nuevo.",
    }
    CHECKSUM_PATH.write_text(json.dumps(checksum, indent=1) + "\n", encoding="utf-8")
    print(f"Guardado: {NPZ_PATH} sha256={checksum['sha256']}", flush=True)


if __name__ == "__main__":
    main()
