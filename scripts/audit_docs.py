"""Check local Markdown assets and known contradictory scientific-status phrases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


def inspect(repo):
    """Return actionable local links and stale claim locations without network writes."""
    root = Path(repo).resolve()
    missing, suspect = [], []
    patterns = [r"CONTRATO-005.{0,100}pending", r"script.{0,100}not yet in the repository",
                r"No phase masks are committed", r"No second implementation",
                r"CONTRATO-003.{0,100}In progress", r"matches the exact solution"]
    for path in root.rglob("*.md"):
        if any(part in {".git", ".venv", "node_modules", "completion-kit"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for destination in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
            split = urlsplit(destination.strip("<>"))
            if split.scheme or not split.path or destination.startswith("#"):
                continue
            target = (path.parent/unquote(split.path)).resolve()
            if not target.exists():
                missing.append(dict(file=str(path.relative_to(root)), target=destination))
        for number, line in enumerate(text.splitlines(), 1):
            if any(re.search(pattern, line, re.I) for pattern in patterns):
                suspect.append(dict(file=str(path.relative_to(root)), line=number, text=line))
    return dict(missing_local_links=missing, status_phrases_to_review=suspect,
                scope="heuristic local audit; scientific consistency requires human review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = inspect(args.repo)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["missing_local_links"] else 0)


if __name__ == "__main__":
    main()
