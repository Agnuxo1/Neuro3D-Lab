"""Write a hash manifest of a frozen protocol for external registration and timestamping.

Upload the resulting JSON (and the files it lists) to an external registry such as OSF
Registries or a Zenodo record before TEST is opened, and optionally timestamp it with
OpenTimestamps ('ots stamp protocol-manifest.json'). A git commit alone is internal:
its date can be rewritten, so it does not prove when a protocol existed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state(repo):
    """Commit and cleanliness of the checkout; None values when git is unavailable."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return dict(head=head, clean=not dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(head=None, clean=None)


def build(files, repo, study_id):
    """Hash every listed file relative to the repository root."""
    repo = Path(repo).resolve()
    entries = {}
    for name in files:
        path = (repo/name).resolve()
        if not path.is_file() or not path.is_relative_to(repo):
            raise FileNotFoundError(f"not a file inside the repository: {name}")
        entries[str(path.relative_to(repo)).replace("\\", "/")] = sha256(path)
    if not entries:
        raise ValueError("list at least one protocol file")
    return dict(study_id=study_id, created_utc=datetime.now(timezone.utc).isoformat(),
                git=git_state(repo), files=entries,
                note="Register externally before TEST is opened; the hash list is what is being registered.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.files, args.repo, args.study_id)
    if manifest["git"]["clean"] is False:
        raise SystemExit("commit the protocol first: the checkout has uncommitted changes")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
