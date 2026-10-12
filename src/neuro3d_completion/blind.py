"""Blind analysis of final predictions: analyse coded models, then unblind under commitment.

Practice borrowed from particle physics (Klein and Roodman, Annu. Rev. Nucl. Part. Sci.
55, 141-163 (2005), DOI 10.1146/annurev.nucl.55.090704.151521). The analyst receives
predictions under random codes, writes and freezes the analysis code, and records its
hash together with the hash of the sealed key. Unblinding verifies both commitments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import shutil

from .evidence import digest_file, write_json_new


def blind(results_dir, output_dir, key_path, models, seeds, predictions_name="predictions.npz"):
    """Copy predictions to coded directories; write the sealed key and its commitment."""
    results_dir, output_dir, key_path = Path(results_dir), Path(output_dir), Path(key_path)
    if key_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("the key must be stored outside the analyst's directory")
    codes, used = {}, set()
    for model in models:
        code = secrets.token_hex(4)
        while code in used:
            code = secrets.token_hex(4)
        used.add(code)
        codes[model] = code
    output_dir.mkdir(parents=True, exist_ok=False)
    for model, code in codes.items():
        for seed in seeds:
            source = results_dir/f"{model}_seed{seed}"/predictions_name
            target = output_dir/f"{code}_seed{seed}"/predictions_name
            target.parent.mkdir(parents=True)
            shutil.copyfile(source, target)
    write_json_new(key_path, dict(codes=codes, seeds=list(seeds)))
    write_json_new(output_dir/"blinding.json", dict(key_sha256=digest_file(key_path), seeds=list(seeds),
                                                   coded_models=sorted(codes.values())))
    return output_dir/"blinding.json"


def freeze_analysis(blinded_dir, analysis_files):
    """Record the analysis code hashes before unblinding; refuses to overwrite."""
    blinded_dir = Path(blinded_dir)
    hashes = {str(Path(p)): digest_file(p) for p in analysis_files}
    if not hashes:
        raise ValueError("at least one analysis file is required")
    write_json_new(blinded_dir/"analysis_frozen.json", dict(analysis_sha256=hashes,
                   blinding_sha256=digest_file(blinded_dir/"blinding.json")))


def unblind(blinded_dir, key_path):
    """Verify the key commitment and the frozen analysis, then return the code mapping."""
    blinded_dir, key_path = Path(blinded_dir), Path(key_path)
    commitment = json.loads((blinded_dir/"blinding.json").read_text(encoding="utf-8"))
    frozen_path = blinded_dir/"analysis_frozen.json"
    if not frozen_path.is_file():
        raise RuntimeError("freeze the analysis before unblinding")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if digest_file(key_path) != commitment["key_sha256"]:
        raise RuntimeError("key does not match the commitment")
    if frozen["blinding_sha256"] != digest_file(blinded_dir/"blinding.json"):
        raise RuntimeError("blinding record changed after the analysis was frozen")
    for path, expected in frozen["analysis_sha256"].items():
        if digest_file(path) != expected:
            raise RuntimeError(f"analysis file changed after freezing: {path}")
    return json.loads(key_path.read_text(encoding="utf-8"))["codes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("blind")
    b.add_argument("--results", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--key", type=Path, required=True)
    b.add_argument("--models", nargs="+", required=True)
    b.add_argument("--seeds", type=int, nargs="+", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--blinded", type=Path, required=True)
    f.add_argument("--analysis", type=Path, nargs="+", required=True)
    u = sub.add_parser("unblind")
    u.add_argument("--blinded", type=Path, required=True)
    u.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "blind":
        print(blind(args.results, args.output, args.key, args.models, args.seeds))
    elif args.command == "freeze":
        freeze_analysis(args.blinded, args.analysis)
    else:
        print(json.dumps(unblind(args.blinded, args.key), indent=2))


if __name__ == "__main__":
    main()
