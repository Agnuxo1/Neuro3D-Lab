"""The Blender add-on ships the NumPy-only pieces of neuro3d_completion under _vendor.

This test fails if that copy drifts from src/neuro3d_completion (guide G-12).
"""
import filecmp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "neuro3d_completion"
VENDOR = ROOT / "blender" / "neuro3d_lab" / "_vendor" / "neuro3d_completion"
FILES = ("__init__.py", "core.py", "bench_adapter.py")


def test_vendored_completion_backend_matches_source():
    assert sorted(p.name for p in VENDOR.glob("*.py")) == sorted(FILES)
    for name in FILES:
        assert filecmp.cmp(SRC / name, VENDOR / name, shallow=False), f"{name} differs from src/neuro3d_completion"
    assert (VENDOR / "LICENSE").is_file()
