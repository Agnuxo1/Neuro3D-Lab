"""Check a closure manifest against real artifact bytes; never infer a Nobel claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SIMULATION_GATES = ("legacy_compatibility", "independent_gradient", "boundary_convergence",
                    "pixel_convergence", "scalar_reference", "uncertainty_budget",
                    "benchmarks", "controls", "paired_statistics", "environment_lock",
                    "data_artifacts", "blender_integration", "documentation_consistency",
                    "manuscript", "independent_replication")
PHYSICAL_GATES = ("vector_model", "fabrication_tolerances", "calibrated_hardware")


def check(path):
    """Require an explicit scope and validated artifacts for every claimed gate."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scope = manifest.get("scope")
    if scope not in {"scalar_simulation", "physical_device"}:
        raise ValueError("scope must be scalar_simulation or physical_device")
    required = SIMULATION_GATES + (PHYSICAL_GATES if scope == "physical_device" else ())
    failures = []
    for name in required:
        gate = manifest.get("gates", {}).get(name, {})
        if gate.get("status") != "passed":
            failures.append(f"{name}: not passed"); continue
        artifact, expected = gate.get("artifact"), gate.get("sha256")
        if not isinstance(artifact, str) or not isinstance(expected, str) or len(expected) != 64:
            failures.append(f"{name}: missing artifact provenance"); continue
        target = (path.parent/Path(artifact)).resolve()
        if not target.is_file():
            failures.append(f"{name}: artifact absent"); continue
        h = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            failures.append(f"{name}: artifact hash mismatch")
    return dict(scope=scope, ready=not failures, failures=failures,
                interpretation="artifact-completeness check; scientific review remains necessary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    result = check(args.manifest)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
