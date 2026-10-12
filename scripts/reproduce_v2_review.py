"""Regenerate the exploratory numbers quoted in Part II of the guide (v2 review).

Each part writes one JSON record (refusing overwrites) into --output. None of these
parts reads TEST or QMNIST50K. Parts and approximate cost on 2 CPU cores:

  roughness   §15.1  phase-step statistics of the historical masks          < 1 s
  screen      §15.2  pixelated screen versus RS at large windows             ~ 30 s
  crosscheck  §15.3  core r=3 versus RS q=3, with and without evanescent waves ~ 10 s
  window_r4   §13    window convergence at refinement 4 (needs ~4 GB RAM)      ~ 30 s
  encoding    §13    E6: data.encode versus the CONTRATO-008 encoding          ~ 5 s
  writers     §17.3  contiguous versus writer-disjoint DEV, writer effects     ~ 10 min
  budget      §19    CPU training throughput versus FFT size                   ~ 1 min
  ranks       §20.2  Jacobian rank of the reachable operator set               ~ 1 s
  proxy       §15.3  full-scale core r=1/3/5 on two writer-disjoint DEV images   ~ 20 s

Results depend on hardware for timings only; numerical values should agree to the
printed precision (the MLP of 'writers' may vary slightly across PyTorch builds).
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from neuro3d_completion.core import DiffractiveNetwork, Optics  # noqa: E402
from neuro3d_completion.evidence import record  # noqa: E402
from neuro3d_completion.validation import relative  # noqa: E402

PHASES = ROOT/"experiments"/"results"/"d2nn_seed0_phases.npz"


def roughness(phases_path=PHASES):
    """Mean |wrapped phase step| between horizontal and vertical neighbours, per layer."""
    with np.load(phases_path, allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    wrap = lambda d: np.angle(np.exp(1j*d))
    random = np.random.default_rng(0).uniform(0, 2*np.pi, phases.shape[1:])
    return dict(horizontal=[float(np.mean(np.abs(wrap(np.diff(p, axis=1))))) for p in phases],
                vertical=[float(np.mean(np.abs(wrap(np.diff(p, axis=0))))) for p in phases],
                uniform_expected=float(np.pi/2),
                uniform_seed0_horizontal=float(np.mean(np.abs(wrap(np.diff(random, axis=1))))))


def screen():
    """§15.2 large-window rows and the zero-phase control."""
    from neuro3d_completion.pixel_reference import asm_at_centres, pixel_centres, screen_function
    from neuro3d_completion.validation import rayleigh_sommerfeld
    n, a, lam, d = 16, 4e-4, 7.5e-4, 0.018
    w = 0.25*n*a
    out = {}
    for label, phases in (("random_seed0", np.random.default_rng(0).uniform(0, 2*np.pi, (n, n))),
                          ("zero_phase", np.zeros((n, n)))):
        source = screen_function(phases, a, w)
        reference = {q: rayleigh_sommerfeld(source, n*a, a/q, pixel_centres(n, a), lam, d, block=16)
                     for q in (9, 27)}
        rows = {}
        settings = ([("none", 1, p) for p in (8, 32, 64, 128)] + [("phase_gradient", 1, p) for p in (32, 64, 128)] +
                    [("none", 3, p) for p in (8, 32, 64)] + [("none", 9, 32)])
        if label == "zero_phase":
            settings = [("none", 1, 8), ("none", 3, 8)]
        for mode, r, p in settings:
            field = asm_at_centres(phases, a, lam, d, w, r, p, mode)
            rows[f"{mode}_r{r}_p{p}"] = {f"vs_rs_q{q}": relative(field, ref) for q, ref in reference.items()}
        out[label] = rows
    return out


def crosscheck():
    """§15.3: n=8, one random layer, d=6 mm, core r=3 versus direct RS q=3."""
    from neuro3d_completion.rs_network import rs_network_fields
    n, a, lam, d = 8, 4e-4, 7.5e-4, 0.006
    rng = np.random.default_rng(1)
    x = np.zeros((1, n, n))
    x[:, n//4:3*n//4, n//4:3*n//4] = rng.uniform(0.2, 1, (1, n//2, n//2))
    u = (x/np.linalg.norm(x))[0]
    phases = np.random.default_rng(3).uniform(0, 2*np.pi, (1, n, n))
    masks = np.zeros((2, n, n)); masks[0, 2:4, 2:4] = 1; masks[1, 4:6, 4:6] = 1
    reference = rs_network_fields(u, phases, a, lam, (d, d), q=3)[0]
    rows = {}
    for p in (8, 16, 32, 64, 128):
        model = DiffractiveNetwork(Optics(n=n, pixel_m=a, wavelength_m=lam, distances_m=(d, d), padding=p,
                                          refinement=3, boundary="aperture"), masks)
        model.phases = phases.copy()
        rows[f"p{p}_propagating_only"] = relative(model.forward(u)["f"][model.roi]*3/a, reference)
        if p >= 64:
            o = model.optics
            f = np.fft.fftfreq(o.m, d=o.sample_pitch_m)
            fx, fy = np.meshgrid(f, f, indexing="xy")
            q2 = lam**-2 - fx**2 - fy**2
            kernel = np.where(q2 > 0, np.exp(2j*np.pi*d*np.sqrt(np.maximum(q2, 0))),
                              np.exp(-2*np.pi*d*np.sqrt(np.maximum(-q2, 0))))
            model.kernels = [kernel, kernel]
            rows[f"p{p}_with_evanescent"] = relative(model.forward(u)["f"][model.roi]*3/a, reference)
    return rows


def window_r4(phases_path=PHASES):
    """Window convergence of the historical masks at refinement 4 (two smooth inputs)."""
    from neuro3d_completion.validation import boundary_convergence
    with np.load(phases_path, allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    optics = Optics(n=128, padding=2, refinement=4, boundary="aperture")
    return boundary_convergence(optics, phases, batch_size=1, paddings=(2, 4, 8))


def encoding(mnist_dir, count=2000):
    """E6: relative L2 between data.encode and the CONTRATO-008 encoding (unit energy both)."""
    import torch
    from torch.nn import functional as F
    from neuro3d_completion.data import encode, read_idx
    x = read_idx(Path(mnist_dir)/"train-images-idx3-ubyte.gz")[:count]
    a = encode(x)
    t = torch.tensor(x/255.0, dtype=torch.float64)[:, None]
    b = F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False).clamp(0, 1)[:, 0].numpy()
    b = b/np.sqrt((b*b).sum(axis=(1, 2)))[:, None, None]
    d = np.linalg.norm((a - b).reshape(len(a), -1), axis=1)
    return dict(count=count, median=float(np.median(d)), p95=float(np.quantile(d, 0.95)), max=float(d.max()))


def _read_idx2(path):
    with gzip.open(path, "rb") as stream:
        data = stream.read()
    return np.frombuffer(data, ">i4", offset=12).reshape(-1, 8).astype(np.int64)


def _mlp(x_train, y_train, seed, epochs=3, batch=64):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    x, y = torch.tensor(x_train), torch.tensor(y_train)
    for _ in range(epochs):
        for index in np.array_split(rng.permutation(len(y)), len(y)//batch):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x[index]), y[index]).backward()
            optimizer.step()
    return model


def _errors(model, x, y):
    import torch
    with torch.no_grad():
        return (model(torch.tensor(x)).argmax(1).numpy() != y).astype(float)


def writers(mnist_dir, qmnist_dir, writer_split_dir, seeds=5):
    """§17.3 measurements; uses only MNIST training images (no TEST, no QMNIST50K)."""
    import torch
    from neuro3d_completion.data import read_idx
    torch.set_num_threads(2)
    images = read_idx(Path(mnist_dir)/"train-images-idx3-ubyte.gz").reshape(60000, -1).astype(np.float32)/255
    labels = read_idx(Path(mnist_dir)/"train-labels-idx1-ubyte.gz").astype(np.int64)
    meta = _read_idx2(Path(qmnist_dir)/"qmnist-train-labels-idx2-int.gz")
    out = {}
    # (a) contiguous DEV (50000-59999) versus writer-disjoint DEV
    writer_train = np.load(Path(writer_split_dir)/"train.npz")
    writer_dev = np.load(Path(writer_split_dir)/"dev.npz")
    flat = lambda d: d["images"].reshape(len(d["labels"]), -1).astype(np.float32)/255
    seen = np.isin(meta[50000:, 2], np.unique(meta[:50000, 2]))
    contiguous, disjoint, seen_err, unseen_err = [], [], [], []
    for seed in range(seeds):
        model = _mlp(images[:50000], labels[:50000], seed)
        wrong = _errors(model, images[50000:], labels[50000:])
        contiguous.append(wrong.mean()); seen_err.append(wrong[seen].mean()); unseen_err.append(wrong[~seen].mean())
        model = _mlp(flat(writer_train), writer_train["labels"].astype(np.int64), seed)
        disjoint.append(_errors(model, flat(writer_dev), writer_dev["labels"]).mean())
    out["dev_error"] = dict(contiguous=contiguous, writer_disjoint=disjoint)
    out["contiguous_seen_unseen"] = dict(seen_images=int(seen.sum()), unseen_images=int((~seen).sum()),
                                         seen_error=seen_err, unseen_error=unseen_err,
                                         seen_series={int(s): int(c) for s, c in zip(*np.unique(meta[50000:, 1][seen], return_counts=True))},
                                         unseen_series={int(s): int(c) for s, c in zip(*np.unique(meta[50000:, 1][~seen], return_counts=True))})
    # (b) series-4 control: held-out writers versus held-out images of seen writers
    control = []
    for seed in range(seeds):
        rng = np.random.default_rng(100 + seed)
        series4 = np.unique(meta[meta[:, 1] == 4, 2]); rng.shuffle(series4)
        unseen_writers, seen_writers = series4[:60], series4[60:]
        unseen_mask = np.isin(meta[:, 2], unseen_writers)
        held = np.zeros(60000, bool)
        for writer in seen_writers:
            index = np.flatnonzero(meta[:, 2] == writer)
            held[rng.choice(index, max(1, len(index)//5), replace=False)] = True
        train = np.flatnonzero(~unseen_mask & ~held)
        eval_unseen, eval_seen = np.flatnonzero(unseen_mask), np.flatnonzero(held)
        k = min(len(eval_unseen), len(eval_seen))
        eval_unseen, eval_seen = rng.choice(eval_unseen, k, replace=False), rng.choice(eval_seen, k, replace=False)
        model = _mlp(images[train], labels[train], seed)
        control.append(dict(seen=_errors(model, images[eval_seen], labels[eval_seen]).mean(),
                            unseen=_errors(model, images[eval_unseen], labels[eval_unseen]).mean(), n_each=int(k)))
    difference = [100*(c["unseen"] - c["seen"]) for c in control]
    out["series4_control"] = dict(runs=control, unseen_minus_seen_points_mean=float(np.mean(difference)),
                                  sd_across_seeds=float(np.std(difference, ddof=1)))
    # (c) design effect of one model's error rate on the writer-disjoint DEV
    groups_of = writer_dev["writer_ids"]
    groups = [np.flatnonzero(groups_of == w) for w in np.unique(groups_of)]
    design = []
    for seed in range(3):
        model = _mlp(flat(writer_train), writer_train["labels"].astype(np.int64), seed)
        wrong = _errors(model, flat(writer_dev), writer_dev["labels"])
        rng = np.random.default_rng(7)
        iid = [wrong[rng.integers(len(wrong), size=len(wrong))].mean() for _ in range(2000)]
        grouped = [wrong[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])].mean()
                   for _ in range(2000)]
        deff = (np.std(grouped)/np.std(iid))**2
        design.append(dict(error=float(wrong.mean()), design_effect=float(deff), effective_n=float(len(wrong)/deff)))
    out["design_effect"] = dict(writers=len(groups), runs=design)
    return out


def budget(protocol_path=ROOT/"completion-configs"/"closure_mnist.json"):
    """§19 table on the current CPU."""
    import json
    from neuro3d_completion.budget import estimate, measure
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    rows = []
    for padding, refinement in ((2, 1), (4, 1), (4, 2), (4, 4)):
        optics = Optics(**{**protocol["optics"], "padding": padding, "refinement": refinement,
                           "distances_m": tuple(protocol["optics"]["distances_m"])})
        measured = measure(optics, "cpu", batch=2, steps=2)
        rows.append(dict(padding=padding, refinement=refinement, measurement=measured,
                         estimate=estimate(protocol, measured["seconds_per_sample"])))
    return rows


def ranks():
    """§20.2: rank of d vec(A) / d phases for n=6, p=2."""
    from neuro3d_completion.expressivity import jacobian_rank
    n = 6
    masks = np.zeros((2, n, n)); masks[0, 1:3, 1:3] = 1; masks[1, 3:5, 3:5] = 1
    out = {}
    for layers, boundary in ((1, "aperture"), (2, "aperture"), (3, "aperture"), (3, "transparent")):
        model = DiffractiveNetwork(Optics(n=n, distances_m=(0.006,)*(layers + 1), padding=2, boundary=boundary),
                                   masks, beta=1.0, seed=layers)
        out[f"L{layers}_{boundary}"] = jacobian_rank(model)
    return out


def proxy(writer_split_dir, phases_path=PHASES):
    """§15.3 full-scale substitute: core scores at r=1, 3, 5 (p=4) for DEV images 0 and 1."""
    from neuro3d_completion.core import detector_masks
    from neuro3d_completion.data import encode
    with np.load(phases_path, allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    with np.load(Path(writer_split_dir)/"dev.npz", allow_pickle=False) as data:
        x, labels = encode(data["images"][[0, 1]]), data["labels"][[0, 1]]
    x = np.pad(x, ((0, 0), (32, 32), (32, 32)))
    scores = {}
    for r in (1, 3, 5):
        model = DiffractiveNetwork(Optics(n=128, padding=4, refinement=r, boundary="aperture"),
                                   detector_masks(128, 4e-4), beta=1.0)
        model.phases = phases.copy()
        scores[r] = model.forward(x)["scores"]
    return dict(r1_vs_r3=relative(scores[1], scores[3]), r3_vs_r5=relative(scores[3], scores[5]),
                labels={f"r{r}": s.argmax(axis=1).tolist() for r, s in scores.items()}, true_labels=labels.tolist(),
                scores={f"r{r}": s.tolist() for r, s in scores.items()})


PARTS = ("roughness", "screen", "crosscheck", "window_r4", "encoding", "writers", "budget", "ranks", "proxy")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=[p for p in PARTS if p != "window_r4"])
    parser.add_argument("--mnist", type=Path, default=Path("datasets/mnist"))
    parser.add_argument("--qmnist", type=Path, default=Path("datasets/qmnist"))
    parser.add_argument("--writer-split", type=Path, default=Path("data/mnist-writer"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calls = dict(roughness=lambda: roughness(), screen=screen, crosscheck=crosscheck, window_r4=lambda: window_r4(),
                 encoding=lambda: encoding(args.mnist), writers=lambda: writers(args.mnist, args.qmnist, args.writer_split),
                 budget=lambda: budget(), ranks=ranks, proxy=lambda: proxy(args.writer_split))
    for part in args.parts:
        start = time.perf_counter()
        values = calls[part]()
        record(args.output/f"{part}.json", f"v2 review reproduction: {part}",
               dict(part=part, seconds=time.perf_counter() - start), values)
        print(args.output/f"{part}.json")


if __name__ == "__main__":
    main()
