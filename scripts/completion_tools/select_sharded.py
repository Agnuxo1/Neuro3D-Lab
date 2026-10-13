"""DEV selection split into one process per grid point, assembled into the same frozen record.

`neuro3d_completion.benchmark.select` trains every grid point of every model in one process.
On a GPU shared through a FIFO queue that is a multi-hour ticket. This tool runs exactly the
same training function (`benchmark.train`, seed 0, full TRAIN, DEV history, last epoch) for one
(model, grid index) per invocation, then assembles a frozen file with the keys that
`benchmark.final` and `benchmark.summarize` read. The winner rule is the one of `select`:
highest last-epoch DEV accuracy, ties broken by the declared grid order. Every shard records
the package code digest; assembly refuses shards produced by different code.

Usage:
  python select_sharded.py plan     --protocol P
  python select_sharded.py shard    --protocol P --shards DIR --model d2nn --index 0 --device cuda
  python select_sharded.py assemble --protocol P --shards DIR --frozen F --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from neuro3d_completion.benchmark import (MODELS, equivalence_gate, load_protocol, read_split, train)
from neuro3d_completion.core import Optics
from neuro3d_completion.evidence import code_digest, digest_file, environment, write_json_new


def grid(protocol, name):
    """The grid of `benchmark.select`, in the same declared order."""
    if name == "d2nn":
        return [dict(beta=beta, lr=lr) for beta in protocol.get("beta_grid", [100, 200, 400])
                for lr in protocol.get("optical_lr_grid", [0.02, 0.05])]
    return [dict(lr=lr) for lr in protocol.get("digital_lr_grid", [.001, .003])]


def deterministic(device):
    """Same global settings as `benchmark.main`."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def shard(protocol_path, shards, name, index, device):
    """Train one grid point with seed 0 and record its DEV history."""
    deterministic(device)
    protocol = load_protocol(protocol_path)
    setting = grid(protocol, name)[index]
    target = Path(shards)/f"{name}_{index}.json"
    if target.exists():
        raise FileExistsError(f"shard already recorded: {target}")
    data_dir = Path(protocol["data_dir"])
    x, y, _ = read_split(data_dir/"train.npz")
    dev_x, dev_y, _ = read_split(data_dir/"dev.npz")
    optics = Optics(**protocol["optics"])
    t0 = time.perf_counter()
    _, history = train(name, optics, setting, 0, x, y, dev_x, dev_y, protocol["epochs"], protocol["batch"], device)
    write_json_new(target, dict(model=name, index=index, setting=setting, dev_accuracy=history[-1]["dev_accuracy"],
                                history=history, device=device, elapsed_s=time.perf_counter() - t0,
                                code_sha256=code_digest(), protocol_file_sha256=digest_file(protocol_path),
                                data_sha256={s: digest_file(data_dir/(s + ".npz")) for s in ("train", "dev")}))


def assemble(protocol_path, shards, output, device):
    """Apply the `select` winner rule to complete shards and freeze the record."""
    deterministic(device)
    protocol = load_protocol(protocol_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError("selection is already frozen")
    code, protocol_hash = code_digest(), digest_file(protocol_path)
    data_dir = Path(protocol["data_dir"])
    selected, selections = {}, {}
    for name in MODELS:
        runs = []
        for index, setting in enumerate(grid(protocol, name)):
            record = json.loads((Path(shards)/f"{name}_{index}.json").read_text(encoding="utf-8"))
            if (record["setting"] != setting or record["code_sha256"] != code or
                    record["protocol_file_sha256"] != protocol_hash or record["device"] != device):
                raise RuntimeError(f"shard {name}_{index} does not match the protocol, code or device")
            for split, value in record["data_sha256"].items():
                if digest_file(data_dir/(split + ".npz")) != value:
                    raise RuntimeError("data changed between shards and assembly")
            runs.append(dict(setting=setting, dev_accuracy=record["dev_accuracy"], history=record["history"]))
        winner = max(range(len(runs)), key=lambda i: (runs[i]["dev_accuracy"], -i))
        selected[name], selections[name] = runs[winner]["setting"], runs
    baselines = MODELS[1:]
    best = max(baselines, key=lambda name: max(r["dev_accuracy"] for r in selections[name]))
    gate = equivalence_gate(Optics(**protocol["optics"]), device)
    write_json_new(output, dict(protocol=protocol, protocol_file_sha256=protocol_hash, selected=selected,
                               selections=selections, best_baseline_on_dev=best,
                               data_sha256={name: digest_file(data_dir/(name + ".npz"))
                                            for name in ("train", "dev", "test")},
                               gate=gate, environment=environment(), selection_device=device,
                               selection_mode="sharded: one process per (model, grid index); training function "
                                              "benchmark.train, seed 0, full TRAIN, last epoch; winner rule of select"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "shard", "assemble"])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--index", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.command == "plan":
        protocol = load_protocol(args.protocol)
        print(json.dumps({name: grid(protocol, name) for name in MODELS}, indent=1))
    elif args.command == "shard":
        if args.shards is None or args.model is None or args.index is None:
            parser.error("--shards, --model and --index are required")
        shard(args.protocol, args.shards, args.model, args.index, args.device)
    else:
        if args.shards is None or args.frozen is None:
            parser.error("--shards and --frozen are required")
        assemble(args.protocol, args.shards, args.frozen, args.device)


if __name__ == "__main__":
    main()
