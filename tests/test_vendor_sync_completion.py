"""The Blender add-on ships the NumPy-only pieces of neuro3d_completion under _vendor.

This test fails if that copy drifts from src/neuro3d_completion (guide G-12).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "neuro3d_completion"
VENDOR = ROOT / "blender" / "neuro3d_lab" / "_vendor" / "neuro3d_completion"
FILES = ("__init__.py", "core.py", "bench_adapter.py")


def test_vendored_completion_backend_matches_source():
    assert sorted(p.name for p in VENDOR.glob("*.py")) == sorted(FILES)
    for name in FILES:
        source = (SRC / name).read_text(encoding="utf-8")
        if name == "bench_adapter.py":
            # Only the import is adapted to the extension namespace; numerical code is identical.
            original = "    from optlab.bench import load_bench"
            assert source.count(original) == 1
            source = source.replace(original, "    from ..optlab.bench import load_bench")
        assert (VENDOR / name).read_text(encoding="utf-8") == source, f"{name} differs from src/neuro3d_completion"
    assert (VENDOR / "LICENSE").is_file()
