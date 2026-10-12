"""Claim register and manuscript language audit.

Every scientific statement that a manuscript, README or talk makes is an entry in a JSON
register with a falsification criterion, a scope, a status and hashed evidence. Refuted
and inconclusive claims stay in the register. The text audit flags strong wording that is
not tied to a supported claim with a tag such as [C:H1-MNIST].
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

STATUSES = {"supported", "refuted", "inconclusive", "pending"}
TYPES = {"simulation", "statistical", "physical", "novelty", "software", "theory"}
STRONG = re.compile(r"\b(first|novel|breakthrough|unprecedented|nobel|revolutionary|state[- ]of[- ]the[- ]art|"
                    r"outperform\w*|superior|exact solution|proves?|guarantee\w*|energy[- ]efficient|"
                    r"speed[- ]of[- ]light|faster than|orders of magnitude)\b", re.I)
TAG = re.compile(r"\[C:([A-Za-z0-9_.-]+)\]")


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_register(path, previous=None):
    """Return a list of problems; an empty list means the register is internally complete."""
    path = Path(path).resolve()
    register = json.loads(path.read_text(encoding="utf-8"))
    claims, problems, seen = register.get("claims", []), [], set()
    for claim in claims:
        cid = claim.get("id")
        if not cid or cid in seen:
            problems.append(f"missing or duplicate id: {cid}")
            continue
        seen.add(cid)
        for field in ("statement", "type", "scope", "falsification", "status"):
            if not claim.get(field):
                problems.append(f"{cid}: missing {field}")
        if claim.get("status") not in STATUSES:
            problems.append(f"{cid}: invalid status")
        if claim.get("type") not in TYPES:
            problems.append(f"{cid}: invalid type")
        evidence = claim.get("evidence", [])
        if claim.get("status") in {"supported", "refuted"} and not evidence:
            problems.append(f"{cid}: a decided claim needs evidence")
        for item in evidence:
            target = (path.parent/item.get("artifact", "")).resolve()
            if not target.is_file():
                problems.append(f"{cid}: evidence absent {item.get('artifact')}")
            elif _sha256(target) != item.get("sha256"):
                problems.append(f"{cid}: evidence hash mismatch {item.get('artifact')}")
        kinds = {item.get("kind") for item in evidence}
        if claim.get("status") == "supported" and claim.get("type") == "novelty" and "prior_art_search" not in kinds:
            problems.append(f"{cid}: novelty requires a recorded prior-art search")
        if claim.get("status") == "supported" and claim.get("type") == "physical" and "calibrated_measurement" not in kinds:
            problems.append(f"{cid}: physical claims require calibrated measurements")
    if previous is not None:
        old = json.loads(Path(previous).read_text(encoding="utf-8"))
        removed = {c.get("id") for c in old.get("claims", [])} - seen
        if removed:
            problems.append(f"claims removed since the previous register: {sorted(removed)}")
    return problems


def audit_text(text_path, register_path):
    """Flag strong wording on lines without a tag to a supported claim."""
    register = json.loads(Path(register_path).read_text(encoding="utf-8"))
    supported = {c["id"] for c in register.get("claims", []) if c.get("status") == "supported"}
    findings = []
    for number, line in enumerate(Path(text_path).read_text(encoding="utf-8").splitlines(), 1):
        words = STRONG.findall(line)
        if not words:
            continue
        tags = set(TAG.findall(line))
        if not tags & supported:
            findings.append(dict(line=number, words=sorted({w.lower() for w in words}), text=line.strip()[:200],
                                 tags=sorted(tags)))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--text", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    problems = check_register(args.register, args.previous)
    findings = {str(p): audit_text(p, args.register) for p in args.text}
    print(json.dumps(dict(register_problems=problems, unsupported_strong_wording=findings), indent=2))
    sys.exit(1 if problems or any(findings.values()) else 0)


if __name__ == "__main__":
    main()
