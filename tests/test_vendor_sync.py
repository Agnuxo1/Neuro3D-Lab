"""The Blender add-on ships a copy of the optlab core under blender/neuro3d_lab/_vendor/optlab.

This test fails if that copy drifts from src/optlab, so the catalogue package never
runs different physics from the repository.
"""
import filecmp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "optlab"
VENDOR = ROOT / "blender" / "neuro3d_lab" / "_vendor" / "optlab"


def _py_files(folder):
    return sorted(p.name for p in folder.glob("*.py"))


def test_vendored_core_matches_source():
    assert _py_files(SRC) == _py_files(VENDOR), "file list differs; re-copy src/optlab into _vendor"
    for name in _py_files(SRC):
        assert filecmp.cmp(SRC / name, VENDOR / name, shallow=False), f"{name} differs from src/optlab"
