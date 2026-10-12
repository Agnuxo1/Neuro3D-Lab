"""Run and retain strict Blender self-test and extension validation evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def run(blender, repo, output):
    """Require every reported step to PASS even if the legacy exit code is zero."""
    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, NEURO3D_LAB_OUT=str(output))
    commands = [
        [blender, "-b", "--factory-startup", "-P", str(repo/"blender/selftest/run_selftest.py")],
        [blender, "--command", "extension", "validate", str(repo/"blender/neuro3d_lab")]]
    results = []
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True, timeout=1800)
        text = result.stdout+"\n"+result.stderr
        (output/f"check_{index}.log").write_text(text, encoding="utf-8")
        accepted = result.returncode == 0
        if index == 0:
            match = re.search(r"SELFTEST_RESULTS (.*?)\nSELFTEST_COUNT", text, re.S)
            if match is None:
                accepted = False
            else:
                steps = json.loads(match.group(1))
                accepted &= bool(steps) and all(status == "PASS" for status in steps.values())
        results.append(dict(command=command, exit_code=result.returncode, accepted=accepted))
    with (output/"summary.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(results=results, passed=all(r["accepted"] for r in results)), stream, indent=2)
    if not all(r["accepted"] for r in results):
        raise RuntimeError("Blender checks incomplete or failed; inspect the retained logs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", default=shutil.which("blender"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.blender:
        parser.error("a Blender executable is required via PATH or --blender")
    run(args.blender, args.repo, args.output)
    print(args.output/"summary.json")


if __name__ == "__main__":
    main()
