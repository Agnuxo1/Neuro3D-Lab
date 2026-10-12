"""Install the guide's additive files into a checkout without replacing legacy code."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def integrate(repository):
    """Copy new modules, tests, protocols, and scripts after collision checks."""
    kit = Path(__file__).resolve().parents[1]
    repo = Path(repository).resolve()
    if not (repo/"src/optlab/propagation.py").is_file() or not (repo/"pyproject.toml").is_file():
        raise ValueError("destination is not an optlab checkout")
    pairs = []
    for source in (kit/"neuro3d_completion").glob("*.py"):
        pairs.append((source, repo/"src/neuro3d_completion"/source.name))
    for source in (kit/"tests").glob("test_*.py"):
        pairs.append((source, repo/"tests"/source.name))
    for source in (kit/"scripts").glob("*.py"):
        pairs.append((source, repo/"scripts"/source.name))
    for source in (kit/"configs").glob("*.json"):
        pairs.append((source, repo/"completion-configs"/source.name))
    for source in (kit/"blender_completion").glob("*.py"):
        pairs.append((source, repo/"blender/neuro3d_lab"/source.name))
    for relative in ("requirements-completion.txt", ".github/workflows/completion.yml"):
        pairs.append((kit/relative, repo/relative))
    if not pairs:
        raise ValueError("completion kit is empty")
    for source, target in pairs:
        if target.exists() and source.read_bytes() != target.read_bytes():
            raise FileExistsError(f"refusing to replace existing {target.relative_to(repo)}")
    for source, target in pairs:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    print(f"Integrated {integrate(args.repo)} additive files")


if __name__ == "__main__":
    main()
