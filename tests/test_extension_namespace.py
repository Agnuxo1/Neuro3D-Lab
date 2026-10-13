"""Bundled imports must work under Blender's namespace without global path changes."""
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

ADDON = Path(__file__).resolve().parents[1] / "blender" / "neuro3d_lab"


def test_shipped_extension_has_no_sys_import_or_absolute_core_import():
    for path in ADDON.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in {"sys", "optlab", "neuro3d_completion"}
                           for alias in node.names), path
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] not in {"sys", "optlab", "neuro3d_completion"}, path


def test_bundled_backend_in_nested_namespace_with_global_name_collisions(tmp_path, monkeypatch):
    package_name = "_extension_namespace_test_vendor"
    spec = importlib.util.spec_from_file_location(package_name, ADDON / "_vendor" / "__init__.py",
                                                 submodule_search_locations=[str(ADDON / "_vendor")])
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, package)
    sentinels = {name: ModuleType(name) for name in ("optlab", "neuro3d_completion")}
    for name, sentinel in sentinels.items():
        monkeypatch.setitem(sys.modules, name, sentinel)
    before = tuple(sys.path)
    try:
        spec.loader.exec_module(package)
        bench = importlib.import_module(f"{package_name}.optlab.bench")
        adapter = importlib.import_module(f"{package_name}.neuro3d_completion.bench_adapter")
        payload = {"schema": "optlab.bench/1", "wavelength_m": 633e-9,
                   "grid": {"n": 8, "pixel_m": 8e-6}, "source": {"z_m": 0},
                   "layers": [], "detector": {"z_m": 1e-4, "regions": []},
                   "output": {"intensity_npy": "intensity.npy"}}
        path = tmp_path / "bench.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        legacy = bench.simulate_bench(path)
        result = adapter.simulate_bench_v2(path, padding=1, boundary="periodic")
        np.testing.assert_allclose(result["intensity"], legacy, rtol=1e-12, atol=1e-12)
        assert tuple(sys.path) == before
        assert all(sys.modules[name] is sentinel for name, sentinel in sentinels.items())
    finally:
        for name in list(sys.modules):
            if name.startswith(package_name + "."):
                del sys.modules[name]
