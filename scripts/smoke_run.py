"""Exercise every benchmark stage on explicitly synthetic data; no scientific claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuro3d_completion.core import Optics
from neuro3d_completion.data import save_split
from neuro3d_completion.evidence import write_json_new
from neuro3d_completion.benchmark import select, final, summarize, MODELS, CONTROLS


def run(output):
    """Create small synthetic fixtures and execute selection, all models, and assembly."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = output/"data"
    rng = np.random.default_rng(2026)
    for split, count, offset in (("train", 30, 0), ("dev", 10, 30), ("test", 20, 0)):
        images = rng.uniform(.02, .1, (count, 8, 8))
        labels = np.arange(count) % 10
        for i, label in enumerate(labels):
            images[i, label % 8, :] += .5
        save_split(data/(split+".npz"), images, labels, np.arange(offset, offset+count))
    protocol = dict(study_id="SYNTHETIC-SOFTWARE-SMOKE", data_dir="data", epochs=1, batch=4,
                    seeds=[0, 1], optics=asdict(Optics(padding=2)), beta_grid=[100],
                    optical_lr_grid=[.02], digital_lr_grid=[.001],
                    hypotheses=dict(mean_accuracy_min=.90, seed_accuracy_min=.88, alpha=.05,
                                    permuted_accuracy_max=.15, untrained_accuracy_max=.20,
                                    no_layer_accuracy_max=.20))
    protocol_path, frozen = output/"protocol.json", output/"selected.json"
    write_json_new(protocol_path, protocol)
    torch.set_num_threads(1)
    select(protocol_path, frozen, "cpu")
    for name in MODELS+CONTROLS:
        for seed in protocol["seeds"]:
            final(frozen, output/"runs", name, seed, "cpu")
    summarize(frozen, output/"runs", output/"summary.json")
    summary = json.loads((output/"summary.json").read_text())
    if len(summary["models"]) != 9:
        raise AssertionError("not all models and controls were assembled")
    return dict(status="PASS", scope="synthetic software smoke, not an optical benchmark",
                models=9, seeds=2, result=str(output/"summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
