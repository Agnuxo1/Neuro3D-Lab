"""Extended closure check: the original gates plus research-excellence gates.

The original scripts/release_check.py is unchanged and still decides the scalar and
physical closures. This script adds gates that a demanding external review would ask
for. Passing it is an artifact-completeness statement, never a claim of impact or prize.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_check import SIMULATION_GATES, PHYSICAL_GATES  # noqa: E402

EXCELLENCE_GATES = ("external_preregistration", "prior_art_review", "power_analysis",
                    "discretisation_order_gci", "rs_direct_reference", "writer_disjoint_dev",
                    "fresh_test_q50k", "writer_grouped_statistics", "blind_analysis",
                    "robustness_dev", "noise_energy_budget", "expressivity_dof",
                    "claims_register", "red_team_review", "ai_assistance_disclosure",
                    "archived_release_doi")
SCOPES = {"scalar_excellence": SIMULATION_GATES + EXCELLENCE_GATES,
          "physical_excellence": SIMULATION_GATES + PHYSICAL_GATES + EXCELLENCE_GATES}


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path):
    """Every required gate must be passed with an existing artifact whose hash matches."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scope = manifest.get("scope")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {sorted(SCOPES)}")
    failures = []
    for name in SCOPES[scope]:
        gate = manifest.get("gates", {}).get(name, {})
        if gate.get("status") != "passed":
            failures.append(f"{name}: {gate.get('status', 'missing')}")
            continue
        artifact, expected = gate.get("artifact"), gate.get("sha256")
        target = (path.parent/str(artifact)).resolve() if isinstance(artifact, str) else None
        if target is None or not target.is_file() or not isinstance(expected, str):
            failures.append(f"{name}: missing artifact provenance")
        elif _sha256(target) != expected:
            failures.append(f"{name}: artifact hash mismatch")
    return dict(scope=scope, ready=not failures, failures=failures,
                interpretation="artifact completeness only; scientific review and external replication decide merit")


def initial_manifest(target, scope="scalar_excellence"):
    """Create a manifest with every gate pending; refuses to overwrite."""
    gates = {name: dict(status="pending", reason="Evidence not yet produced and reviewed",
                        next_action="Run the corresponding protocol step and record the artifact hash")
             for name in SCOPES["physical_excellence"]}
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(dict(scope=scope, gates=gates), stream, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--init", action="store_true", help="create a pending manifest at this path")
    parser.add_argument("--scope", choices=sorted(SCOPES), default="scalar_excellence")
    args = parser.parse_args()
    if args.init:
        initial_manifest(args.manifest, args.scope)
        print(args.manifest)
        return
    result = check(args.manifest)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
