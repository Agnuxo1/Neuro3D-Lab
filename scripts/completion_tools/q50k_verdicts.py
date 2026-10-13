"""Registered MNIST verdicts of CONTRATO-013 on a writer-annotated split (TEST10K or QMNIST50K).

For every model: mean, sample SD and minimum accuracy across the frozen seeds. For the D2NN
against each of the five digital models: seed-0 exact McNemar, Holm over the five, and the
crossed bootstrap over paired seeds and NIST writers (2,000 draws). H2 on this split holds when
the D2NN mean exceeds the DEV-selected primary baseline mean, the seed-0 difference is positive,
the Holm-adjusted p < alpha, and the writer-grouped 95 % interval lies above zero. The code was
written and committed with the pre-registration, before any final run existed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuro3d_completion.benchmark import CONTROLS, MODELS
from neuro3d_completion.evidence import crossed_bootstrap, digest_file, holm, paired_comparison, write_json_new


def verdicts(frozen_path, results_dir, split_path, predictions_name):
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    seeds, alpha = frozen["protocol"]["seeds"], frozen["protocol"]["hypotheses"]["alpha"]
    with np.load(split_path, allow_pickle=False) as data:
        labels, ids, writers, series = data["labels"], data["sample_ids"], data["writer_ids"], data["hsf_series"]
    stacks, models = {}, {}
    for name in MODELS + CONTROLS:
        rows = []
        for seed in seeds:
            with np.load(Path(results_dir)/f"{name}_seed{seed}"/predictions_name, allow_pickle=False) as data:
                if not np.array_equal(data["sample_ids"], ids) or not np.array_equal(data["labels"], labels):
                    raise RuntimeError(f"{name} seed {seed}: predictions not aligned with the split")
                rows.append(data["predictions"])
        stacks[name] = np.asarray(rows)
        accuracy = (stacks[name] == labels).mean(axis=1)
        models[name] = dict(per_seed=accuracy.tolist(), mean=float(accuracy.mean()),
                            sample_sd=float(accuracy.std(ddof=1)), minimum=float(accuracy.min()),
                            per_series={int(s): float((stacks[name][:, series == s] == labels[series == s]).mean())
                                        for s in np.unique(series)})
    comparisons = {name: paired_comparison(labels, stacks["d2nn"][0], stacks[name][0]) for name in MODELS[1:]}
    for name, p in zip(MODELS[1:], holm([comparisons[name]["p_two_sided"] for name in MODELS[1:]])):
        comparisons[name]["holm_p"] = p
        comparisons[name]["writer_grouped_bootstrap"] = crossed_bootstrap(labels, stacks["d2nn"], stacks[name],
                                                                          group_ids=writers)
    primary = frozen["best_baseline_on_dev"]
    test = comparisons[primary]
    h2 = (models["d2nn"]["mean"] > models[primary]["mean"] and test["difference"] > 0 and
          test["holm_p"] < alpha and test["writer_grouped_bootstrap"]["percentile_95"][0] > 0)
    return dict(split=str(split_path), split_sha256=digest_file(split_path), samples=int(len(labels)),
                writers=int(len(np.unique(writers))), seeds=seeds, models=models, seed0_comparisons=comparisons,
                primary_baseline=primary, H2_writer_grouped=bool(h2), frozen_sha256=digest_file(frozen_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--predictions", default="predictions.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verdicts(args.frozen, args.results, args.split, args.predictions)
    write_json_new(args.output, result)
    print(json.dumps(dict(H2_writer_grouped=result["H2_writer_grouped"], primary=result["primary_baseline"],
                          d2nn=result["models"]["d2nn"]["mean"])))


if __name__ == "__main__":
    main()
