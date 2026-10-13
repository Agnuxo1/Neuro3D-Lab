"""Check, before any post-selection step, that code and data are those frozen by the selection.

`benchmark final` checks this itself; `benchmark summarize`, `qmnist evaluate` and `qmnist summarize`
do not. CONTRATO-013 requires this check before each of them, and before reading QMNIST50K it also
requires the registered SHA-256 of `q50k.npz`. A JSON record is written for every check.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from neuro3d_completion.evidence import code_digest, digest_file

REGISTERED_SPLIT_SHA256 = {
    "mnist-writer/q50k": "fb1ff5b2e522655135af7a14744d7c21e6f5fb44b99889fc5ece4dc2fce624f4",
}


def verify(frozen_path, q50k=None):
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    problems = []
    if frozen["environment"]["code_sha256"] != code_digest():
        problems.append("package code differs from the frozen selection")
    data_dir = Path(frozen["protocol"]["data_dir"])
    for split, expected in frozen["data_sha256"].items():
        if digest_file(data_dir/(split + ".npz")) != expected:
            problems.append(f"split {split} differs from the frozen selection")
    if q50k is not None:
        if digest_file(q50k) != REGISTERED_SPLIT_SHA256["mnist-writer/q50k"]:
            problems.append("q50k.npz differs from the registered SHA-256")
    return dict(frozen=str(frozen_path), frozen_sha256=digest_file(frozen_path), q50k=str(q50k) if q50k else None,
                checked_utc=datetime.now(timezone.utc).isoformat(), problems=problems, passed=not problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--q50k", type=Path)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.frozen, args.q50k)
    with args.record.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
