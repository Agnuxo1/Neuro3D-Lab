"""DEV-only selection, frozen configurations, matched-seed evaluation, and summaries.

This is a new protocol. It must not overwrite any historic experiment or claim
that a previously examined test set is fresh. GPU use is explicit in every record.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from .core import Optics, DiffractiveNetwork, detector_masks
from .data import encode
from .evidence import (code_digest, digest_file, environment, holm, paired_comparison,
                       crossed_bootstrap, reserve_test, wilson, write_json_new)
from .torch_backend import TorchD2NN, digital_model, check_equivalence

MODELS = ("d2nn", "linear", "mlp12", "mlp128", "cnn", "quadratic12")
CONTROLS = ("permuted", "untrained", "no_layers")


def read_split(path):
    """Load one admitted split and preserve its sample IDs for paired statistics."""
    with np.load(path, allow_pickle=False) as data:
        x, y, ids = data["images"], data["labels"], data["sample_ids"]
    if y.shape != (len(x),) or ids.shape != y.shape or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("malformed split")
    if np.any(y < 0) or np.any(y > 9) or len(np.unique(ids)) != len(ids):
        raise ValueError("invalid classes or duplicate sample IDs")
    return encode(x), y, ids


def tensor_inputs(images, name, optics, device):
    """Use identical resized unit-energy images for optics and digital baselines."""
    x = torch.as_tensor(images, dtype=torch.float64, device=device)
    if name == "d2nn":
        off = (optics.n-x.shape[-1])//2
        if off < 0 or 2*off+x.shape[-1] != optics.n:
            raise ValueError("image and physical support do not align")
        return F.pad(x, (off, off, off, off))
    return x[:, None]


def make_model(name, optics, setting, seed, device):
    """Initialize each final seed independently using the frozen model definition."""
    torch.manual_seed(seed)
    if name == "d2nn":
        model = TorchD2NN(DiffractiveNetwork(optics, detector_masks(optics.n, optics.pixel_m),
                                           beta=setting["beta"], seed=seed))
    else:
        model = digital_model(name)
    return model.to(device)


def restore_parameters(model, path):
    """Restore only admitted trainable parameters; deterministic FFT buffers are rebuilt."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    if set(state) != set(parameters):
        raise ValueError("saved parameter names do not match the model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if state[name].shape != parameter.shape or not torch.isfinite(state[name]).all():
                raise ValueError("invalid saved parameter tensor")
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    return model


def predict(model, images, name, optics, device, batch):
    """Evaluate fixed weights without using labels or modifying the optimizer."""
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            scores = model(tensor_inputs(images[start:start+batch], name, optics, device))
            if not torch.isfinite(scores).all():
                raise FloatingPointError("nonfinite predictions")
            values.append(scores.argmax(dim=1).cpu().numpy())
    return np.concatenate(values)


def train(name, optics, setting, seed, x, y, dev_x, dev_y, epochs, batch, device):
    """Train on TRAIN; report DEV history; use the final epoch as registered."""
    model = make_model(name, optics, setting, seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=setting["lr"])
    rng, history = np.random.default_rng(seed+1000), []
    for epoch in range(epochs):
        model.train()
        order, loss_sum = rng.permutation(len(y)), 0.0
        for start in range(0, len(y), batch):
            ids = order[start:start+batch]
            scores = model(tensor_inputs(x[ids], name, optics, device))
            labels = torch.as_tensor(y[ids], dtype=torch.int64, device=device)
            loss = F.cross_entropy(scores, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())*len(ids)
        dev_pred = predict(model, dev_x, name, optics, device, batch)
        history.append(dict(epoch=epoch+1, loss=loss_sum/len(y), dev_accuracy=float(np.mean(dev_pred == dev_y))))
    return model, history


def load_protocol(path):
    """Read a fixed protocol, resolving dataset paths against its own directory."""
    path = Path(path).resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    for key in ("study_id", "data_dir", "optics", "epochs", "batch", "seeds", "hypotheses"):
        if key not in protocol:
            raise ValueError(f"missing protocol key: {key}")
    protocol["data_dir"] = str((path.parent/Path(protocol["data_dir"])).resolve())
    seeds = protocol["seeds"]
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds) or seeds[0] != 0 or
            any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds)):
        raise ValueError("at least two distinct nonnegative paired seeds starting with seed 0 are required")
    if protocol["epochs"] < 1 or protocol["batch"] < 1:
        raise ValueError("positive training budget required")
    return protocol


def equivalence_gate(optics, device):
    """Reject an inconsistent accelerated backend before any training."""
    reference = DiffractiveNetwork(optics, detector_masks(optics.n, optics.pixel_m), seed=0)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, optics.n, optics.n)) + 1j*rng.normal(size=(2, optics.n, optics.n))
    x /= np.linalg.norm(x, axis=(1, 2))[:, None, None]
    result = check_equivalence(reference, x, np.array([0, 3]), device)
    if not result["pass"]:
        raise RuntimeError(f"accelerated equivalence gate failed: {result}")
    return result


def select(protocol_path, output, device):
    """Select settings on DEV, then freeze all models and data/code fingerprints."""
    protocol = load_protocol(protocol_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError("selection is already frozen")
    data_dir = Path(protocol["data_dir"])
    x, y, _ = read_split(data_dir/"train.npz")
    dev_x, dev_y, _ = read_split(data_dir/"dev.npz")
    optics = Optics(**protocol["optics"])
    gate = equivalence_gate(optics, device)
    selected, selections = {}, {}
    for name in MODELS:
        grid = ([dict(beta=beta, lr=lr) for beta in protocol.get("beta_grid", [100, 200, 400])
                 for lr in protocol.get("optical_lr_grid", [0.02, 0.05])] if name == "d2nn"
                else [dict(lr=lr) for lr in protocol.get("digital_lr_grid", [.001, .003])])
        runs = []
        for setting in grid:
            _, history = train(name, optics, setting, 0, x, y, dev_x, dev_y,
                               protocol["epochs"], protocol["batch"], device)
            runs.append(dict(setting=setting, dev_accuracy=history[-1]["dev_accuracy"], history=history))
        winner = max(range(len(runs)), key=lambda i: (runs[i]["dev_accuracy"], -i))
        selected[name], selections[name] = runs[winner]["setting"], runs
    baselines = MODELS[1:]
    best = max(baselines, key=lambda name: max(r["dev_accuracy"] for r in selections[name]))
    write_json_new(output, dict(protocol=protocol, protocol_file_sha256=digest_file(protocol_path),
                               selected=selected, selections=selections, best_baseline_on_dev=best,
                               data_sha256={name: digest_file(data_dir/(name+".npz"))
                                            for name in ("train", "dev", "test")},
                               gate=gate, environment=environment(), selection_device=device))


def final(frozen_path, output_dir, name, seed, device):
    """Train frozen settings, reserve TEST, and persist weights and all predictions."""
    frozen_path, output_dir = Path(frozen_path), Path(output_dir)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol = frozen["protocol"]
    if seed not in protocol["seeds"] or name not in MODELS+CONTROLS:
        raise ValueError("model or seed outside the frozen protocol")
    if frozen["environment"]["code_sha256"] != code_digest():
        raise RuntimeError("code changed after freezing; amend the protocol before selection, not after TEST")
    data_dir = Path(protocol["data_dir"])
    for split, expected in frozen["data_sha256"].items():
        if digest_file(data_dir/(split+".npz")) != expected:
            raise RuntimeError("data changed after freezing")
    run_dir = output_dir/f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_new(run_dir/"started.json", dict(frozen_sha256=digest_file(frozen_path), model=name, seed=seed))
    t0 = time.perf_counter()
    x, y, _ = read_split(data_dir/"train.npz")
    dev_x, dev_y, _ = read_split(data_dir/"dev.npz")
    optics = Optics(**protocol["optics"])
    equivalence_gate(optics, device)
    model_name = name if name in MODELS else "d2nn"
    setting, history = frozen["selected"][model_name], []
    if name == "permuted":
        y = np.random.default_rng(100+seed).permutation(y)
    if name == "no_layers":
        optics = Optics(**{**protocol["optics"], "distances_m": [sum(optics.distances_m)]})
    if name in {"untrained", "no_layers"}:
        model = make_model(model_name, optics, setting, seed, device)
    else:
        model, history = train(model_name, optics, setting, seed, x, y, dev_x, dev_y,
                               protocol["epochs"], protocol["batch"], device)
    torch.save({key: value.detach().cpu().clone() for key, value in model.named_parameters()},
               run_dir/"weights.pt")
    write_json_new(run_dir/"architecture.json", dict(optics=asdict(optics), model=model_name,
                                                   setting=setting, seed=seed, dtype="float64"))
    if model_name == "d2nn":
        with (run_dir/"phases.npz").open("xb") as stream:
            np.savez_compressed(stream, phases=model.phases.detach().cpu().numpy(),
                                n=optics.n, pixel_m=optics.pixel_m, wavelength_m=optics.wavelength_m,
                                beta=setting["beta"], seed=seed, distances_m=optics.distances_m,
                                padding=optics.padding, refinement=optics.refinement,
                                boundary=optics.boundary)
    # The immutable reservation happens before opening any TEST arrays.
    reserve_test(output_dir/"test_ledger", protocol["study_id"], name, seed,
                 data_dir/"test.npz", frozen_path)
    test_x, test_y, test_ids = read_split(data_dir/"test.npz")
    pred = predict(model, test_x, model_name, optics, device, protocol["batch"])
    with (run_dir/"predictions.npz").open("xb") as stream:
        np.savez_compressed(stream, predictions=pred, labels=test_y, sample_ids=test_ids)
    correct = int(np.sum(pred == test_y))
    write_json_new(run_dir/"result.json", dict(model=name, seed=seed,
                    frozen_sha256=digest_file(frozen_path), history=history, test_count=len(pred),
                    test_accuracy=correct/len(pred), wilson_95=wilson(correct, len(pred)),
                    trainable_parameters=sum(p.numel() for p in model.parameters()), device=device,
                    elapsed_s=time.perf_counter()-t0, environment=environment(),
                    predictions_sha256=digest_file(run_dir/"predictions.npz"),
                    weights_sha256=digest_file(run_dir/"weights.pt")))


def summarize(frozen_path, output_dir, output):
    """Assemble every declared seed/model, preserving paired labels and provenance."""
    frozen_path, output_dir = Path(frozen_path), Path(output_dir)
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    protocol, predictions, summaries, labels, ids = frozen["protocol"], {}, {}, None, None
    for name in MODELS+CONTROLS:
        rows, stack = [], []
        for seed in protocol["seeds"]:
            run = output_dir/f"{name}_seed{seed}"
            result = json.loads((run/"result.json").read_text(encoding="utf-8"))
            if (result["frozen_sha256"] != digest_file(frozen_path) or
                    result["predictions_sha256"] != digest_file(run/"predictions.npz")):
                raise RuntimeError("result provenance mismatch")
            with np.load(run/"predictions.npz", allow_pickle=False) as data:
                y, sample_ids, pred = data["labels"], data["sample_ids"], data["predictions"]
            if labels is None:
                labels, ids = y, sample_ids
            if not np.array_equal(y, labels) or not np.array_equal(ids, sample_ids):
                raise RuntimeError("models do not share the identical paired TEST set")
            rows.append(result["test_accuracy"]); stack.append(pred)
        predictions[name] = np.asarray(stack)
        summaries[name] = dict(per_seed=rows, mean=float(np.mean(rows)),
                               sample_sd=float(np.std(rows, ddof=1)), minimum=float(np.min(rows)))
    comparisons = {name: paired_comparison(labels, predictions["d2nn"][0], predictions[name][0])
                   for name in MODELS[1:]}
    adjusted = holm([comparisons[name]["p_two_sided"] for name in MODELS[1:]])
    for name, p in zip(MODELS[1:], adjusted):
        comparisons[name]["holm_p"] = p
        comparisons[name]["crossed_bootstrap"] = crossed_bootstrap(labels, predictions["d2nn"], predictions[name])
    primary = frozen["best_baseline_on_dev"]
    hypotheses = protocol["hypotheses"]
    test = comparisons[primary]
    verdicts = dict(
        H1=summaries["d2nn"]["mean"] >= hypotheses["mean_accuracy_min"] and
           summaries["d2nn"]["minimum"] >= hypotheses["seed_accuracy_min"],
        H2=summaries["d2nn"]["mean"] > summaries[primary]["mean"] and
           test["difference"] > 0 and test["holm_p"] < hypotheses["alpha"] and
           test["crossed_bootstrap"]["percentile_95"][0] > 0,
        C1=summaries["permuted"]["mean"] <= hypotheses["permuted_accuracy_max"],
        C2=summaries["untrained"]["mean"] <= hypotheses["untrained_accuracy_max"],
        C3=summaries["no_layers"]["mean"] <= hypotheses["no_layer_accuracy_max"])
    write_json_new(output, dict(frozen_sha256=digest_file(frozen_path), models=summaries,
                               seed0_comparisons=comparisons, primary_baseline=primary,
                               sample_count=len(labels), seeds=protocol["seeds"],
                               study_id=protocol["study_id"],
                               verdicts=verdicts, scope="scalar simulation under the frozen protocol"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select", "final", "summarize"])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("completion_results"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=MODELS+CONTROLS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.command == "select":
        if args.protocol is None:
            parser.error("--protocol is required for selection")
        select(args.protocol, args.frozen, args.device)
    elif args.command == "final":
        if args.model is None:
            parser.error("--model is required for final")
        final(args.frozen, args.output_dir, args.model, args.seed, args.device)
    else:
        if args.output is None:
            parser.error("--output is required for summarize")
        summarize(args.frozen, args.output_dir, args.output)


if __name__ == "__main__":
    main()
