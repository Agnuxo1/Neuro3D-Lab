"""CONTRATO-014: discretisation fidelity of trained D2NN masks (TRAIN/DEV only, never TEST).

Question: do phase masks trained with one numerical sample per physical pixel (refinement 1)
keep their DEV accuracy when the same physical pixels are simulated as piecewise-constant
elements with tied subpixels (refinement 2, 4, 8, 16) and against direct Rayleigh-Sommerfeld?

Stages (each refuses to overwrite its output):
  train     one (refinement, padding, seed) on the first TRAIN_SUBSET images of TRAIN
  evaluate  the saved phases of one run under another (refinement, padding) on DEV
  rs        direct Rayleigh-Sommerfeld scores of the saved phases on a few DEV images
  summarize all records into one JSON with the registered verdicts F1 and F2
The registered design and thresholds are in docs/CONTRACTS/CONTRATO-014-fidelity.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from neuro3d_completion.benchmark import make_model, predict, read_split, train
from neuro3d_completion.core import Optics, detector_masks
from neuro3d_completion.evidence import code_digest, digest_file, environment, write_json_new
from neuro3d_completion.guards import refuse_reserved_split

BASE = dict(n=128, pixel_m=4e-4, wavelength_m=7.5e-4, distances_m=(0.018,)*4, boundary="aperture",
            filter_mode="none")
SETTING = dict(beta=100.0, lr=0.02)
TRAIN_SUBSET, EPOCHS, BATCH, MONITOR = 10000, 10, 16, 512
EVAL_LIMIT_R16 = 2000


def deterministic(device):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def run_name(refinement, padding, seed):
    return f"train_r{refinement}_p{padding}_seed{seed}"


def optics(refinement, padding):
    return Optics(**BASE, padding=padding, refinement=refinement)


def train_stage(data_dir, out_dir, refinement, padding, seed, device):
    """Train with the registered fixed setting; save phases and the training history."""
    deterministic(device)
    run = Path(out_dir)/run_name(refinement, padding, seed)
    run.mkdir(parents=True, exist_ok=False)
    train_path, dev_path = refuse_reserved_split(Path(data_dir)/"train.npz"), refuse_reserved_split(Path(data_dir)/"dev.npz")
    x, y, ids = read_split(train_path)
    dev_x, dev_y, _ = read_split(dev_path)
    o = optics(refinement, padding)
    t0 = time.perf_counter()
    model, history = train("d2nn", o, SETTING, seed, x[:TRAIN_SUBSET], y[:TRAIN_SUBSET],
                           dev_x[:MONITOR], dev_y[:MONITOR], EPOCHS, BATCH, device)
    phases = model.phases.detach().cpu().numpy()
    with (run/"phases.npz").open("xb") as stream:
        np.savez_compressed(stream, phases=phases, n=o.n, pixel_m=o.pixel_m, wavelength_m=o.wavelength_m,
                            distances_m=o.distances_m, padding=o.padding, refinement=o.refinement,
                            boundary=o.boundary, beta=SETTING["beta"], seed=seed)
    write_json_new(run/"architecture.json", dict(optics=dict(o.__dict__), model="d2nn", setting=SETTING, seed=seed))
    write_json_new(run/"train.json", dict(refinement=refinement, padding=padding, seed=seed, setting=SETTING,
                                          train_subset=TRAIN_SUBSET, train_ids_first_last=[int(ids[0]), int(ids[TRAIN_SUBSET-1])],
                                          epochs=EPOCHS, batch=BATCH, monitor_dev_images=MONITOR, history=history,
                                          elapsed_s=time.perf_counter() - t0, device=device,
                                          phases_sha256=digest_file(run/"phases.npz"),
                                          train_sha256=digest_file(train_path), dev_sha256=digest_file(dev_path),
                                          code_sha256=code_digest(), environment=environment()))


def lean_model(phases, o, beta, device):
    """TorchD2NN with kernels, masks and aperture built directly on the device.

    The standard constructor builds every buffer in NumPy first, which needs about 6 GB of host
    RAM at m = 4096. The formulas are those of core.DiffractiveNetwork (propagating-only transfer
    function, no filter, tied-subpixel detector masks); `lean_check` verifies identity at m = 2048.
    """
    from torch import nn
    from neuro3d_completion.torch_backend import TorchD2NN
    if o.filter_mode != "none" or o.boundary != "aperture":
        raise ValueError("lean model implements the unfiltered aperture convention only")
    model = TorchD2NN.__new__(TorchD2NN)
    nn.Module.__init__(model)
    model.optics, model.beta = o, float(beta)
    model.phases = nn.Parameter(torch.as_tensor(phases, dtype=torch.float64, device=device).clone())
    freq = torch.fft.fftfreq(o.m, d=o.sample_pitch_m, dtype=torch.float64, device=device)
    fx, fy = torch.meshgrid(freq, freq, indexing="xy")
    q2 = o.wavelength_m**-2 - fx**2 - fy**2
    q = torch.sqrt(torch.clamp(q2, min=0))
    kernels = torch.stack([torch.where(q2 > 0, torch.exp(2j*np.pi*d*q), torch.zeros((), dtype=torch.complex128,
                                                                                       device=device))
                           for d in o.distances_m])
    active, off = o.n*o.refinement, (o.m - o.n*o.refinement)//2
    physical = torch.as_tensor(detector_masks(o.n, o.pixel_m), dtype=torch.float64, device=device)
    masks = torch.zeros((len(physical), o.m, o.m), dtype=torch.float64, device=device)
    masks[:, off:off+active, off:off+active] = physical.repeat_interleave(o.refinement, 1).repeat_interleave(o.refinement, 2)
    aperture = torch.zeros((o.m, o.m), dtype=torch.float64, device=device)
    aperture[off:off+active, off:off+active] = 1
    model.register_buffer("kernels", kernels)
    model.register_buffer("masks", masks)
    model.register_buffer("aperture", aperture)
    model.offset = off
    return model


def run_setting(run):
    """The (beta, lr) the phases were trained with: architecture.json of the run (fidelity or benchmark)."""
    architecture = Path(run)/"architecture.json"
    if not architecture.is_file():
        raise FileNotFoundError(f"{architecture} is required to know beta")
    stored = json.loads(architecture.read_text(encoding="utf-8"))
    if stored.get("model") != "d2nn":
        raise ValueError("only D2NN phases can be re-discretised")
    return dict(stored["setting"])


def load_model(run, refinement, padding, device, lean=None):
    """Rebuild a Torch D2NN for the evaluation optics and load the trained physical phases."""
    with np.load(Path(run)/"phases.npz", allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
        seed = int(data["seed"])
    setting = run_setting(run)
    o = optics(refinement, padding)
    if lean is None:
        lean = o.m >= 4096
    if lean:
        return lean_model(phases, o, setting["beta"], device), o
    model = make_model("d2nn", o, setting, seed, device)
    with torch.no_grad():
        model.phases.copy_(torch.as_tensor(phases, dtype=torch.float64, device=device))
    return model, o


def _scores(model, o, x, device):
    model.eval()
    out = []
    batch = 4 if o.m >= 4096 else BATCH  # memory only; scores do not depend on the batch
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.as_tensor(x[start:start+batch], dtype=torch.float64, device=device)
            off = (o.n - xb.shape[-1])//2
            out.append(model(torch.nn.functional.pad(xb, (off, off, off, off))).cpu().numpy())
    return np.concatenate(out)


def lean_check(data_dir, run, device, refinement=8, padding=2, count=32):
    """Identity of the lean and the standard constructor on DEV images (recorded once)."""
    deterministic(device)
    target = Path(run)/f"lean_check_r{refinement}_p{padding}.json"
    x, _, _ = read_split(refuse_reserved_split(Path(data_dir)/"dev.npz"))
    standard, o = load_model(run, refinement, padding, device, lean=False)
    a = _scores(standard, o, x[:count], device)
    del standard
    torch.cuda.empty_cache() if device == "cuda" else None
    lean, _ = load_model(run, refinement, padding, device, lean=True)
    b = _scores(lean, o, x[:count], device)
    value = float(np.max(np.abs(a - b))/np.max(np.abs(a)))
    write_json_new(target, dict(refinement=refinement, padding=padding, images=count,
                                max_relative_score_difference=value, passed=value <= 1e-10))
    if value > 1e-10:
        raise RuntimeError(f"lean constructor differs from the standard one: {value}")


def evaluate_stage(data_dir, run, refinement, padding, device, limit=None):
    """DEV accuracy and per-sample scores of fixed phases under one evaluation discretisation."""
    deterministic(device)
    run = Path(run)
    target = run/f"eval_r{refinement}_p{padding}.json"
    if target.exists():
        raise FileExistsError(target)
    dev_path = refuse_reserved_split(Path(data_dir)/"dev.npz")
    x, y, ids = read_split(dev_path)
    if limit is None and refinement >= 16:
        limit = EVAL_LIMIT_R16
    if limit is not None:
        x, y, ids = x[:limit], y[:limit], ids[:limit]
    if optics(refinement, padding).m >= 4096:
        checks = list(Path(run).parent.glob("*/lean_check_r8_p2.json"))
        if not checks or not all(json.loads(c.read_text(encoding="utf-8"))["passed"] for c in checks):
            raise RuntimeError("the lean constructor must pass its recorded check before any m >= 4096 evaluation")
    model, o = load_model(run, refinement, padding, device)
    t0 = time.perf_counter()
    scores = _scores(model, o, x, device)
    pred = scores.argmax(axis=1)
    with (run/f"eval_r{refinement}_p{padding}_scores.npz").open("xb") as stream:
        np.savez_compressed(stream, scores=scores, labels=y, sample_ids=ids)
    write_json_new(target, dict(eval_refinement=refinement, eval_padding=padding, samples=len(y),
                                dev_accuracy=float(np.mean(pred == y)), elapsed_s=time.perf_counter() - t0,
                                device=device, dev_sha256=digest_file(dev_path),
                                scores_sha256=digest_file(run/f"eval_r{refinement}_p{padding}_scores.npz")))


def rs_stage(data_dir, run, indices, quadratures, device, block):
    """Direct RS scores of the trained phases (absorbing exterior) on a few DEV images."""
    from neuro3d_completion.data import encode
    from neuro3d_completion.rs_network import rs_network_fields, rs_scores
    run = Path(run)
    target = run/"rs_direct.json"
    if target.exists():
        raise FileExistsError(target)
    dev_path = refuse_reserved_split(Path(data_dir)/"dev.npz")
    with np.load(dev_path, allow_pickle=False) as data:
        images, labels, ids = data["images"][indices], data["labels"][indices], data["sample_ids"][indices]
    x = encode(images)
    x = np.pad(x, ((0, 0), (32, 32), (32, 32)))
    with np.load(run/"phases.npz", allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    masks = detector_masks(BASE["n"], BASE["pixel_m"])
    result = dict(indices=list(map(int, indices)), sample_ids=ids.tolist(), labels=labels.tolist(), rs={})
    for q in quadratures:
        t0 = time.perf_counter()
        fields = rs_network_fields(x, phases, BASE["pixel_m"], BASE["wavelength_m"], BASE["distances_m"], q, block, device)
        scores = rs_scores(fields, masks, BASE["pixel_m"], q, beta=run_setting(run)["beta"])
        result["rs"][f"q{q}"] = dict(scores=scores.tolist(), labels=scores.argmax(axis=1).tolist(),
                                     elapsed_s=time.perf_counter() - t0)
    result.update(device=device, dev_sha256=digest_file(dev_path), phases_sha256=digest_file(run/"phases.npz"),
                  scope="direct scalar RS1 summation, piecewise-constant pixels, absorbing exterior; DEV images only")
    write_json_new(target, result)


def _relative(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b)/np.linalg.norm(b))


def summarize(out_dir, output):
    """Collect every record; apply the registered verdicts F1 and F2."""
    out_dir = Path(out_dir)
    table, rs_rows = [], []
    for run in sorted(p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("train_")):
        meta = json.loads((run/"train.json").read_text(encoding="utf-8"))
        for record in sorted(run.glob("eval_r*_p*.json")):
            e = json.loads(record.read_text(encoding="utf-8"))
            table.append(dict(train_refinement=meta["refinement"], train_padding=meta["padding"], seed=meta["seed"],
                              eval_refinement=e["eval_refinement"], eval_padding=e["eval_padding"],
                              samples=e["samples"], dev_accuracy=e["dev_accuracy"]))
        if (run/"rs_direct.json").is_file():
            rs = json.loads((run/"rs_direct.json").read_text(encoding="utf-8"))
            with np.load(run/f"eval_r{meta['refinement']}_p{meta['padding']}_scores.npz", allow_pickle=False) as data:
                own = dict(zip(data["sample_ids"].tolist(), data["scores"]))
            core = np.asarray([own[i] for i in rs["sample_ids"]])
            row = dict(train_refinement=meta["refinement"], train_padding=meta["padding"], seed=meta["seed"],
                       images=len(rs["sample_ids"]))
            for key, value in rs["rs"].items():
                row[f"core_train_discretisation_vs_rs_{key}"] = _relative(core, value["scores"])
                row[f"label_agreement_{key}"] = float(np.mean(core.argmax(axis=1) == np.asarray(value["labels"])))
            if "q1" in rs["rs"] and "q3" in rs["rs"]:
                row["rs_q1_vs_q3"] = _relative(rs["rs"]["q1"]["scores"], rs["rs"]["q3"]["scores"])
            rs_rows.append(row)

    def mean_acc(tr, tp, er, ep):
        values = [r["dev_accuracy"] for r in table if (r["train_refinement"], r["train_padding"],
                                                       r["eval_refinement"], r["eval_padding"]) == (tr, tp, er, ep)]
        return (float(np.mean(values)), len(values)) if values else (None, 0)

    f1_rows = {}
    for tp in (2, 4):
        own, n_own = mean_acc(1, tp, 1, tp)
        fine, n_fine = mean_acc(1, tp, 8, 2)
        if own is not None and fine is not None:
            f1_rows[f"train_r1_p{tp}"] = dict(own=own, eval_r8_p2=fine, drop=own - fine, seeds=min(n_own, n_fine))
    # Robustness line (descriptive): the same drop measured against refinement 16 on its DEV subset.
    r16_rows = {}
    for run in sorted(out_dir.glob("train_r1_p*_seed*")):
        fine = run/"eval_r16_p2_scores.npz"
        meta = json.loads((run/"train.json").read_text(encoding="utf-8"))
        own = run/f"eval_r1_p{meta['padding']}_scores.npz"
        if fine.is_file() and own.is_file():
            with np.load(fine, allow_pickle=False) as a, np.load(own, allow_pickle=False) as b:
                k = len(a["labels"])
                r16_rows[run.name] = dict(own_subset=float(np.mean(b["scores"][:k].argmax(1) == b["labels"][:k])),
                                          eval_r16_subset=float(np.mean(a["scores"].argmax(1) == a["labels"])),
                                          images=k)
    # F2 compares refinement 8 and 16 on the same DEV images (the first EVAL_LIMIT_R16).
    pairs = []
    for run in sorted(out_dir.glob("train_r8_p2_seed*")):
        fine, coarse = run/"eval_r16_p2_scores.npz", run/"eval_r8_p2_scores.npz"
        if fine.is_file() and coarse.is_file():
            with np.load(fine, allow_pickle=False) as a, np.load(coarse, allow_pickle=False) as b:
                k = len(a["labels"])
                if not np.array_equal(a["sample_ids"], b["sample_ids"][:k]):
                    raise RuntimeError("refinement 8 and 16 evaluations are not aligned")
                pairs.append((float(np.mean(b["scores"][:k].argmax(1) == b["labels"][:k])),
                              float(np.mean(a["scores"].argmax(1) == a["labels"])), k))
    r8 = float(np.mean([p[0] for p in pairs])) if pairs else None
    r16 = float(np.mean([p[1] for p in pairs])) if pairs else None
    complete_f1 = set(f1_rows) == {"train_r1_p2", "train_r1_p4"} and all(v["seeds"] == 3 for v in f1_rows.values())
    verdicts = dict(
        F1=bool(all(v["drop"] > 0.05 for v in f1_rows.values())) if complete_f1 else None,
        F2=bool(abs(r8 - r16) <= 0.01) if len(pairs) == 3 else None)
    write_json_new(output, dict(table=table, rs_direct=rs_rows, f1=f1_rows, f1_robustness_r16=r16_rows,
                                f2=dict(train_r8_eval_r8_subset=r8, train_r8_eval_r16_subset=r16, seeds=len(pairs),
                                        subset_images=pairs[0][2] if pairs else None),
                                verdicts=verdicts,
                                scope="DEV only; scalar thin-element model; refinements tie subpixels to one phase per physical pixel"))
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "evaluate", "lean-check", "rs", "summarize"])
    parser.add_argument("--data", type=Path, default=Path("data/mnist-writer"))
    parser.add_argument("--out", type=Path, default=Path("completion_results/fidelity"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--refinement", type=int)
    parser.add_argument("--padding", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--quadratures", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        train_stage(args.data, args.out, args.refinement, args.padding, args.seed, args.device)
    elif args.command == "evaluate":
        evaluate_stage(args.data, args.run, args.refinement, args.padding, args.device, args.limit)
    elif args.command == "lean-check":
        lean_check(args.data, args.run, args.device)
    elif args.command == "rs":
        rs_stage(args.data, args.run, args.indices, args.quadratures, args.device, args.block)
    else:
        print(json.dumps(summarize(args.out, args.output)))


if __name__ == "__main__":
    main()
