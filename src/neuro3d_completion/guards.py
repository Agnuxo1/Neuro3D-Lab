"""Shared safeguards for the v2 exploratory tools.

refuse_reserved_split: exploratory tools must never read TEST or a fresh test set. The
check uses both the file name and, when present, the manifest.json written next to the
splits by data.prepare / qmnist.prepare (split_sha256), so a renamed copy of a reserved
split is still refused.

frozen_optics: rebuild the exact optics of a frozen model from architecture.json (written
by benchmark.final, includes filter_mode); otherwise from the physical fields stored in
the phases NPZ plus explicit numerical settings.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .core import Optics
from .evidence import digest_file

RESERVED_TOKENS = ("test", "q50k", "fresh")


def _reserved(name):
    return any(token in name.lower() for token in RESERVED_TOKENS)


def refuse_reserved_split(path):
    """Raise ValueError if 'path' is, by name or by recorded hash, a reserved evaluation split."""
    path = Path(path)
    if _reserved(path.name):
        raise ValueError(f"exploratory tools must not read a reserved split: {path.name}")
    manifest = path.parent/"manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        hashes = payload.get("values", payload).get("split_sha256", {})
        reserved = {value for key, value in hashes.items() if _reserved(key)}
        if reserved and digest_file(path) in reserved:
            raise ValueError(f"{path.name} has the hash of a reserved split in {manifest}")
    return path


def frozen_optics(phases_path, padding=2, refinement=1, boundary="aperture", filter_mode="none"):
    """Optics for frozen phases; architecture.json takes precedence over the arguments."""
    phases_path = Path(phases_path)
    architecture = phases_path.parent/"architecture.json"
    if architecture.is_file():
        stored = json.loads(architecture.read_text(encoding="utf-8"))["optics"]
        return Optics(**{**stored, "distances_m": tuple(stored["distances_m"])}), "architecture.json"
    with np.load(phases_path, allow_pickle=False) as data:
        phases = np.asarray(data["phases"])
        physical = dict(n=int(phases.shape[-1]))
        for key in ("pixel_m", "wavelength_m"):
            if key in data.files:
                physical[key] = float(data[key])
        if "distances_m" in data.files:
            physical["distances_m"] = tuple(float(d) for d in np.atleast_1d(data["distances_m"]))
        stored_numerics = {key: data[key].item() for key in ("padding", "refinement", "boundary") if key in data.files}
    numerics = dict(padding=padding, refinement=refinement, boundary=boundary, filter_mode=filter_mode)
    numerics.update({k: (str(v) if k == "boundary" else int(v)) for k, v in stored_numerics.items()})
    optics = Optics(**physical, **numerics)
    if optics.layers != len(phases):
        raise ValueError("stored distances do not match the number of phase layers")
    return optics, "phases.npz fields plus explicit numerical settings"
