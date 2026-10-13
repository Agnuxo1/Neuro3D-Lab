# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless self-test of the versioned Run Bench callback (completion guide, phase G).

Run from anywhere (no absolute paths):
  blender -b --factory-startup -P blender/selftest/run_completion_selftest.py
Output goes to <out>/completion_selftest (<out> = $NEURO3D_LAB_OUT or <repo parent>/.cognition/neuro3d-lab).
Checks: the legacy path without a sidecar, the bundled neuro3d_completion backend, the finite-support
path with numerical_config.json and its provenance, identity of the periodic p = 1 backend with the
legacy simulation, import of the new intensity, and a deliberate failure. Exit 0 only if every step PASSES.
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # blender/ folder

import bpy  # noqa: E402
import numpy as np  # noqa: E402

N, PIX, WL = 256, 8e-6, 6.33e-7
results, state = {}, {}


def step(name):
    def deco(fn):
        try:
            r = fn()
            results[name] = r if isinstance(r, str) else "PASS"
        except Exception:
            results[name] = "FAIL: " + traceback.format_exc(limit=3).replace("\n", " | ")
    return deco


bpy.ops.wm.read_factory_settings(use_empty=True)
import neuro3d_lab  # noqa: E402
from neuro3d_lab import bench_io  # noqa: E402

OUT = os.path.join(bench_io.out_root(), "completion_selftest")
SIDECAR = os.path.join(OUT, "numerical_config.json")


@step("register_and_export")
def _():
    neuro3d_lab.register()
    bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(SIDECAR):
        raise RuntimeError("stale sidecar from an earlier run; use a fresh output folder")
    bpy.context.scene.optlab_output_dir = OUT
    assert bpy.ops.optlab.export_bench() == {"FINISHED"}
    state["json"] = os.path.join(OUT, "bench.json")
    assert os.path.isfile(state["json"])


@step("legacy_path_without_sidecar")
def _():
    assert bpy.ops.optlab.run_bench() == {"FINISHED"}
    path = bpy.context.scene.optlab_intensity_path
    assert os.path.normpath(path) == os.path.normpath(os.path.join(OUT, "detector_intensity.npy")), path
    state["legacy"] = np.load(path)
    assert state["legacy"].shape == (N, N) and np.isfinite(state["legacy"]).all()


@step("bundled_backend")
def _():
    import neuro3d_completion
    from neuro3d_completion import bench_adapter
    origin = os.path.normpath(neuro3d_completion.__file__)
    assert os.path.normpath(bench_io.VENDOR_DIR) in origin, origin
    state["adapter"] = bench_adapter


@step("periodic_identity_with_legacy")
def _():
    v2 = state["adapter"].simulate_bench_v2(state["json"], padding=1, refinement=1, boundary="periodic")
    rel = float(np.linalg.norm(v2["intensity"] - state["legacy"])/np.linalg.norm(state["legacy"]))
    state["identity"] = rel
    assert rel <= 1e-12, rel


@step("finite_support_with_sidecar")
def _():
    settings = {"padding": 2, "refinement": 1, "boundary": "aperture", "filter_mode": "none"}
    with open(SIDECAR, "x", encoding="utf-8") as f:
        json.dump(settings, f)
    assert bpy.ops.optlab.run_bench() == {"FINISHED"}
    path = bpy.context.scene.optlab_intensity_path
    run = os.path.dirname(path)
    assert os.path.basename(os.path.dirname(run)) == "finite_support_runs", path
    provenance = json.load(open(os.path.join(run, "provenance.json")))
    assert provenance["settings"] == settings and provenance["backend"] == "neuro3d_completion/v1"
    intensity = np.load(path)
    direct = state["adapter"].simulate_bench_v2(state["json"], **settings)["intensity"]
    assert intensity.shape == (N, N) and np.isfinite(intensity).all()
    assert np.allclose(intensity, direct, rtol=1e-12, atol=0)
    state["v2_path"] = path


@step("import_new_intensity")
def _():
    img = bench_io.import_intensity(state["v2_path"])
    assert tuple(img.size) == (N, N)


@step("deliberate_failure_bad_sidecar")
def _():
    os.replace(SIDECAR, SIDECAR + ".used")
    with open(SIDECAR, "x", encoding="utf-8") as f:
        json.dump({"padding": 2, "refinement": 1, "boundary": "aperture", "filter_mode": "none", "extra": 1}, f)
    before = bpy.context.scene.optlab_intensity_path
    try:  # an operator that reports ERROR raises RuntimeError when called from Python
        outcome = bpy.ops.optlab.run_bench()
    except RuntimeError as exc:
        outcome, state["expected_error"] = {"CANCELLED"}, str(exc)
    assert outcome == {"CANCELLED"}, outcome
    assert bpy.context.scene.optlab_intensity_path == before


@step("unregister")
def _():
    neuro3d_lab.unregister()
    assert not hasattr(bpy.types.Scene, "optlab_grid_n")


print("SELFTEST_RESULTS " + json.dumps(results, indent=1))
print("PERIODIC_IDENTITY " + json.dumps(state.get("identity")))
n_pass = sum(1 for v in results.values() if v == "PASS")
print(f"SELFTEST_COUNT {n_pass}/{len(results)} PASS")
sys.exit(0 if results and all(v == "PASS" for v in results.values()) else 1)
