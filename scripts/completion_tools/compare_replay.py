"""Compare JSON outputs regenerated in a detached replay checkout with the committed historical files.

The historical files are never overwritten: the replay runs in a separate worktree at the
same commit, and this script only reads both copies and reports numeric differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


def leaves(value, prefix=""):
    """Yield (path, value) for every scalar leaf of a JSON document."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def compare(historical, replayed):
    """Maximum absolute and relative numeric difference, plus non-numeric mismatches."""
    a, b = dict(leaves(historical)), dict(leaves(replayed))
    max_abs, max_rel, mismatches = 0.0, 0.0, []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if not (math.isfinite(x) and math.isfinite(y)):
                if x != y:
                    mismatches.append(key)
                continue
            d = abs(x - y)
            max_abs = max(max_abs, d)
            if x != 0:
                max_rel = max(max_rel, d/abs(x))
        elif x != y:
            mismatches.append(key)
    return dict(numeric_leaves=sum(isinstance(v, (int, float)) for v in a.values()),
                max_abs_difference=max_abs, max_rel_difference=max_rel,
                non_numeric_mismatches=mismatches[:50], mismatch_count=len(mismatches))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.run(["git", "-C", str(args.replay), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    report = dict(replay_commit=commit, files={})
    for name in args.files:
        historical = subprocess.run(["git", "-C", str(args.replay), "show", f"HEAD:{name}"], capture_output=True,
                                    check=True).stdout
        replayed = (args.replay/name).read_bytes()
        report["files"][name] = dict(
            historical_sha256=hashlib.sha256(historical).hexdigest(),
            replayed_sha256=hashlib.sha256(replayed).hexdigest(),
            byte_identical=historical == replayed,
            **compare(json.loads(historical), json.loads(replayed)))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: {kk: v[kk] for kk in ("byte_identical", "max_abs_difference", "mismatch_count")}
                      for k, v in report["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
