"""Registered check W1 of CONTRATO-013 (DEV only) and the secondary refinement line.

Reads the per-sample DEV scores that `fidelity_study.py evaluate` wrote for the seed-0 D2NN of a
final benchmark run at two discretisations, and reports the relative L2 change of the scores and
the fraction of changed DEV labels. W1: padding 4 -> 8 at refinement 1 must change at most 0.5 %
of labels and at most 1e-2 of the scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuro3d_completion.evidence import write_json_new


def compare(run, reference, other):
    with np.load(Path(run)/f"eval_{reference}_scores.npz", allow_pickle=False) as a, \
            np.load(Path(run)/f"eval_{other}_scores.npz", allow_pickle=False) as b:
        k = min(len(a["labels"]), len(b["labels"]))
        if not np.array_equal(a["sample_ids"][:k], b["sample_ids"][:k]):
            raise RuntimeError("evaluations are not aligned")
        sa, sb, y = a["scores"][:k], b["scores"][:k], a["labels"][:k]
    return dict(reference=reference, other=other, images=int(k),
                score_relative_L2=float(np.linalg.norm(sb - sa)/np.linalg.norm(sa)),
                changed_label_fraction=float(np.mean(sa.argmax(1) != sb.argmax(1))),
                accuracy_reference=float(np.mean(sa.argmax(1) == y)), accuracy_other=float(np.mean(sb.argmax(1) == y)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--others", nargs="*", default=["r2_p2", "r4_p2", "r8_p2"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    w1 = compare(args.run, "r1_p4", "r1_p8")
    w1["passed"] = w1["changed_label_fraction"] <= 0.005 and w1["score_relative_L2"] <= 1e-2
    refinement = [compare(args.run, "r1_p4", other) for other in args.others
                  if (args.run/f"eval_{other}_scores.npz").is_file()]
    write_json_new(args.output, dict(run=str(args.run), W1=w1, refinement_line=refinement,
                                     scope="DEV only; seed-0 D2NN of the frozen benchmark; refinement line descriptive"))
    print(json.dumps(w1))


if __name__ == "__main__":
    main()
