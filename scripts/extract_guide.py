"""Extract complete source files embedded in the Markdown guide without overwrites."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

PATTERN = re.compile(r"<!-- BEGIN_FILE ([^\n]+) -->\n```[^\n]*\n(.*?)\n```\n<!-- END_FILE -->", re.S)


def extract(markdown_path, destination):
    """Validate all embedded paths before writing any source file."""
    root = Path(destination).resolve()
    text = Path(markdown_path).read_text(encoding="utf-8")
    matches = PATTERN.findall(text)
    if not matches:
        raise ValueError("no complete embedded files found")
    targets, names = [], set()
    for name, content in matches:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in names:
            raise ValueError("unsafe or duplicate embedded path")
        target = (root/path).resolve()
        if not target.is_relative_to(root) or target.exists():
            raise FileExistsError(f"refusing to overwrite or escape destination: {name}")
        names.add(name); targets.append((target, content))
    for target, content in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content+"\n")
    return len(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("guide", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(f"Extracted {extract(args.guide, args.destination)} files")


if __name__ == "__main__":
    main()
