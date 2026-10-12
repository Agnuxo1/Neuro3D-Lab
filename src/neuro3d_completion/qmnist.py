"""Writer-aware MNIST splits and a fresh confirmatory test set from QMNIST.

QMNIST (Yadav and Bottou, "Cold Case: The Lost MNIST Digits", NeurIPS 2019,
arXiv 1905.10498; data at github.com/facebookresearch/qmnist, BSD-style licence)
traces every MNIST digit to its NIST writer and reconstructs the 50,000 MNIST test
digits that were never released. Facts verified on 2026-10-12 with the files hashed below:

* QMNIST train labels (column 0) equal MNIST train labels; QMNIST test rows 0-9999
  equal MNIST test labels. Images are reconstructions and are not byte-identical.
* The contiguous MNIST DEV split (train 50000-59999) shares 251 writers with TRAIN.
* MNIST TRAIN+DEV writers and QMNIST50K writers (test rows 10000-59999) are disjoint;
  QMNIST50K shares 250 writers with the MNIST TEST10K and contains NIST HSF series 1
  (11,956 digits), a series absent from MNIST train and test (series 0 and 4 only).

This module (1) splits the MNIST training set into writer-disjoint TRAIN/DEV, stratified
by HSF series, (2) attaches writer IDs to the official MNIST TEST for writer-grouped
bootstrap, and (3) exports QMNIST50K as a fresh test with writer and series IDs.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import struct
import numpy as np

from .data import HASHES as MNIST_HASHES, FILES as MNIST_FILES, read_idx, save_split
from .evidence import crossed_bootstrap, digest_file, paired_comparison, record, reserve_test, wilson, write_json_new

# SHA-256 of the official files, computed on 2026-10-12 from
# https://raw.githubusercontent.com/facebookresearch/qmnist/master/<file>
QMNIST_HASHES = {
    "qmnist-train-labels-idx2-int.gz": "2c05dc77f6b916b38e455e97ab129a42a444f3dbef09b278a366f82904e0dd9f",
    "qmnist-test-images-idx3-ubyte.gz": "43fc22bf7498b8fc98de98369d72f752d0deabc280a43a7bcc364ab19e57b375",
    "qmnist-test-labels-idx2-int.gz": "9fbcbe594c3766fdf4f0b15c5165dc0d1e57ac604e01422608bb72c906030d06",
}
COLUMNS = ("class", "hsf_series", "writer_id", "writer_digit_index", "nist_class_code",
           "nist_global_index", "duplicate", "unused")


def read_idx2_int(path):
    """Read a QMNIST extended-label file: magic 0x0C02, rows, columns, big-endian int32."""
    with gzip.open(path, "rb") as stream:
        data = stream.read()
    if len(data) < 12:
        raise ValueError("truncated idx2-int header")
    magic, rows, columns = struct.unpack(">III", data[:12])
    if magic != 3074 or columns != 8 or len(data) != 12 + 4*rows*columns:
        raise ValueError("unexpected idx2-int format")
    return np.frombuffer(data, ">i4", offset=12).reshape(rows, columns).astype(np.int64)


def _verify(directory, names, hashes):
    for name in names:
        if digest_file(Path(directory)/name) != hashes[name]:
            raise ValueError(f"SHA-256 mismatch in {name}; nothing has been admitted")


def writer_split(writers, series, dev_fraction=1/6, seed=2026):
    """Boolean DEV mask: whole writers, chosen per HSF series to approach dev_fraction."""
    writers, series = np.asarray(writers), np.asarray(series)
    if writers.shape != series.shape or not 0 < dev_fraction < 1:
        raise ValueError("aligned writers/series and 0 < dev_fraction < 1 are required")
    rng = np.random.default_rng(seed)
    dev = np.zeros(len(writers), dtype=bool)
    for s in np.unique(series):
        in_series = series == s
        ids, counts = np.unique(writers[in_series], return_counts=True)
        target, taken = dev_fraction*in_series.sum(), 0
        for index in rng.permutation(len(ids)):
            if taken >= target:
                break
            dev |= in_series & (writers == ids[index])
            taken += counts[index]
    return dev


def prepare(mnist_dir, qmnist_dir, output, dev_fraction=1/6, seed=2026, mnist_hashes=None, qmnist_hashes=None):
    """Create train/dev/test/q50k NPZ files with sample, writer and series IDs, plus a manifest."""
    mnist_dir, qmnist_dir, output = Path(mnist_dir), Path(qmnist_dir), Path(output)
    mh = dict(zip(MNIST_FILES, (mnist_hashes or MNIST_HASHES["mnist"])))
    qh = qmnist_hashes or QMNIST_HASHES
    _verify(mnist_dir, MNIST_FILES, mh)
    _verify(qmnist_dir, list(qh), qh)
    train_x, train_y, test_x, test_y = (read_idx(mnist_dir/name) for name in MNIST_FILES)
    q_train = read_idx2_int(qmnist_dir/"qmnist-train-labels-idx2-int.gz")
    q_test = read_idx2_int(qmnist_dir/"qmnist-test-labels-idx2-int.gz")
    q_images = read_idx(qmnist_dir/"qmnist-test-images-idx3-ubyte.gz")
    if (len(q_train) != len(train_y) or not np.array_equal(q_train[:, 0], train_y) or
            not np.array_equal(q_test[:len(test_y), 0], test_y) or len(q_test) != len(q_images)):
        raise ValueError("QMNIST labels do not correspond to the admitted MNIST files")
    dev = writer_split(q_train[:, 2], q_train[:, 1], dev_fraction, seed)
    ids = np.arange(len(train_y))
    fresh = slice(len(test_y), len(q_test))
    output.mkdir(parents=True, exist_ok=True)
    for name, x, y, sid, meta in (("train", train_x[~dev], train_y[~dev], ids[~dev], q_train[~dev]),
                                  ("dev", train_x[dev], train_y[dev], ids[dev], q_train[dev]),
                                  ("test", test_x, test_y, np.arange(len(test_y)), q_test[:len(test_y)]),
                                  ("q50k", q_images[fresh], q_test[fresh, 0], np.arange(len(q_test))[fresh],
                                   q_test[fresh])):
        path = output/(name + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, images=x, labels=np.asarray(y, np.int64), sample_ids=np.asarray(sid, np.int64),
                                writer_ids=meta[:, 2], hsf_series=meta[:, 1])
    overlap = len(set(q_train[~dev, 2]) & set(q_train[dev, 2]))
    record(output/"manifest.json", "writer-aware MNIST split with QMNIST50K",
           dict(dev_fraction=dev_fraction, seed=seed),
           dict(counts=dict(train=int((~dev).sum()), dev=int(dev.sum()), test=len(test_y), q50k=len(q_test) - len(test_y)),
                train_dev_shared_writers=overlap,
                train_q50k_shared_writers=len(set(q_train[:, 2]) & set(q_test[fresh, 2])),
                q50k_series_counts={int(s): int(c) for s, c in zip(*np.unique(q_test[fresh, 1], return_counts=True))},
                split_sha256={n: digest_file(output/(n + ".npz")) for n in ("train", "dev", "test", "q50k")},
                source_sha256={**mh, **qh}))
    if overlap:
        raise RuntimeError("writer-disjoint split failed")


def _load_model(run_dir, frozen, device):
    from .benchmark import make_model, restore_parameters
    from .core import Optics
    architecture = json.loads((run_dir/"architecture.json").read_text(encoding="utf-8"))
    optics = Optics(**{**architecture["optics"], "distances_m": tuple(architecture["optics"]["distances_m"])})
    model = make_model(architecture["model"], optics, architecture["setting"], architecture["seed"], device)
    return restore_parameters(model, run_dir/"weights.pt"), optics, architecture["model"]


def fresh_evaluation(frozen_path, results_dir, q50k_path, device="cpu"):
    """Evaluate every frozen final run once on QMNIST50K after reserving that use in the ledger."""
    from .benchmark import MODELS, CONTROLS, predict
    from .data import encode
    frozen_path, results_dir = Path(frozen_path), Path(results_dir)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol = frozen["protocol"]
    runs = [(name, seed, results_dir/f"{name}_seed{seed}") for name in MODELS + CONTROLS for seed in protocol["seeds"]]
    missing = [str(run) for _, _, run in runs if not (run/"weights.pt").is_file() or not (run/"result.json").is_file()]
    if missing:
        raise FileNotFoundError(f"complete every frozen final run before the fresh test: {missing[:3]}")
    # Every use is reserved before the fresh images are opened; a later failure keeps the reservations.
    for name, seed, _ in runs:
        reserve_test(results_dir/"q50k_ledger", protocol["study_id"] + ":Q50K", name, seed, q50k_path, frozen_path)
    with np.load(q50k_path, allow_pickle=False) as data:
        x, y, sid = encode(data["images"]), data["labels"], data["sample_ids"]
    for name, seed, run in runs:
        model, optics, model_name = _load_model(run, frozen, device)
        pred = predict(model, x, model_name, optics, device, protocol["batch"])
        with (run/"predictions_q50k.npz").open("xb") as stream:
            np.savez_compressed(stream, predictions=pred, labels=y, sample_ids=sid)
        write_json_new(run/"result_q50k.json", dict(model=name, seed=seed, accuracy=float(np.mean(pred == y)),
                                                    wilson_95=wilson(int(np.sum(pred == y)), len(y)),
                                                    predictions_sha256=digest_file(run/"predictions_q50k.npz")))


def writer_grouped_summary(frozen_path, results_dir, split_path, predictions_name, output):
    """Accuracy by HSF series and writer-grouped crossed bootstrap of D2NN minus each baseline."""
    from .benchmark import MODELS
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    with np.load(split_path, allow_pickle=False) as data:
        labels, sid, writers, series = data["labels"], data["sample_ids"], data["writer_ids"], data["hsf_series"]
    stacks = {}
    for name in MODELS:
        rows = []
        for seed in frozen["protocol"]["seeds"]:
            with np.load(Path(results_dir)/f"{name}_seed{seed}"/predictions_name, allow_pickle=False) as data:
                if not np.array_equal(data["sample_ids"], sid):
                    raise RuntimeError("predictions and split are not aligned")
                rows.append(data["predictions"])
        stacks[name] = np.asarray(rows)
    result = dict(per_series={}, comparisons={}, writers=int(len(np.unique(writers))))
    for name, stack in stacks.items():
        result["per_series"][name] = {int(s): float(np.mean(stack[:, series == s] == labels[series == s]))
                                      for s in np.unique(series)}
    for name in MODELS[1:]:
        result["comparisons"][name] = dict(
            seed0=paired_comparison(labels, stacks["d2nn"][0], stacks[name][0]),
            writer_grouped_bootstrap=crossed_bootstrap(labels, stacks["d2nn"], stacks[name], group_ids=writers))
    write_json_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--mnist", type=Path, required=True)
    p.add_argument("--qmnist", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    e = sub.add_parser("evaluate")
    e.add_argument("--frozen", type=Path, required=True)
    e.add_argument("--results", type=Path, required=True)
    e.add_argument("--q50k", type=Path, required=True)
    e.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    s = sub.add_parser("summarize")
    s.add_argument("--frozen", type=Path, required=True)
    s.add_argument("--results", type=Path, required=True)
    s.add_argument("--split", type=Path, required=True)
    s.add_argument("--predictions", default="predictions.npz")
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    if args.command == "prepare":
        prepare(args.mnist, args.qmnist, args.output, seed=args.seed)
    elif args.command == "evaluate":
        fresh_evaluation(args.frozen, args.results, args.q50k, args.device)
    else:
        writer_grouped_summary(args.frozen, args.results, args.split, args.predictions, args.output)


if __name__ == "__main__":
    main()
