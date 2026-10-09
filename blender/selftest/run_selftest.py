"""Headless self-test: blender -b --factory-startup -P run_selftest.py"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # blender/ folder

import bpy  # noqa: E402
import numpy as np  # noqa: E402

OUT = "D:/PROJECTS/.cognition/neuro3d-lab/selftest"
OPTLAB_SRC = "D:/PROJECTS/Neuro3D-Lab/src"
N, PIX, WL = 64, 8e-6, 6.33e-7
Z = {"LAYER_1": 0.02, "LAYER_2": 0.04, "DETECTOR": 0.06}
results = {}
state = {}


def step(name):
    """Run the decorated function immediately and record PASS/FAIL/PENDIENTE."""
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


@step("register")
def _():
    neuro3d_lab.register()
    assert hasattr(bpy.ops.optlab, "export_bench")
    assert bpy.context.scene.optlab_grid_n == 64


@step("build")
def _():
    bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)
    names = {o.name for o in bpy.context.scene.objects}
    assert names == {"SOURCE", "LAYER_1", "LAYER_2", "DETECTOR"}, names


@step("export")
def _():
    os.makedirs(OUT, exist_ok=True)
    bpy.context.scene.optlab_output_dir = OUT
    assert bpy.ops.optlab.export_bench() == {"FINISHED"}
    state["json"] = os.path.join(OUT, "bench.json")
    assert os.path.isfile(state["json"])


@step("trace")
def _():
    hits = bench_io.trace_optical_path((0, 0, 0), (0, 0, 1))
    got = {h["object"]: h["distance_m"] for h in hits}
    for k, z in Z.items():
        assert k in got, (k, hits)
        assert abs(got[k] - z) < 1e-6, (k, got[k], z)


@step("schema")
def _():
    b = json.load(open(state["json"]))
    assert b["schema"] == "optlab.bench/1"
    for k in ("wavelength_m", "grid", "source", "layers", "detector", "output"):
        assert k in b, k
    assert b["grid"]["n"] == N and b["grid"]["pixel_m"] == PIX
    zs = [b["source"]["z_m"]] + [l["z_m"] for l in b["layers"]] + [b["detector"]["z_m"]]
    assert all(a < c for a, c in zip(zs, zs[1:])), zs
    assert abs(zs[-1] - 0.06) < 1e-9
    for l in b["layers"]:
        for k in ("name", "n_medium", "n_material", "kappa", "height_npy", "height_max_m"):
            assert k in l, k
        h = np.load(os.path.join(OUT, l["height_npy"]))
        assert h.shape == (N, N) and np.isfinite(h).all() and h.min() >= 0 and h.max() <= 1
    assert len(b["detector"]["regions"]) == 3
    for r in b["detector"]["regions"]:
        assert set(r) == {"label", "cx_m", "cy_m", "w_m", "h_m"}
    assert b["output"]["intensity_npy"]


@step("simulate")
def _():
    if not os.path.isfile(os.path.join(OPTLAB_SRC, "optlab", "bench.py")):
        return "PENDIENTE"
    sys.path.insert(0, OPTLAB_SRC)
    from optlab.bench import simulate_bench
    out = np.asarray(simulate_bench(state["json"]))
    assert out.shape == (N, N) and np.isfinite(out).all(), out.shape
    np.save(os.path.join(OUT, "detector_intensity.npy"), out)


@step("import")
def _():
    rng = np.random.default_rng(1)
    arr = rng.random((N, N)) * 7.0
    arr[3, 5] = 7.5  # asymmetric marker to test orientation
    p = os.path.join(OUT, "test_intensity.npy")
    np.save(p, arr)
    for _i in range(2):  # twice: idempotent
        img = bench_io.import_intensity(p)
    assert tuple(img.size) == (N, N)
    back = bench_io._image_to_array(img) * bpy.context.scene["optlab_intensity_scale"]
    assert np.abs(back - arr).max() < 1e-6, np.abs(back - arr).max()
    n_img = sum(1 for i in bpy.data.images if i.name.startswith("optlab_detector_intensity"))
    assert n_img == 1, n_img
    assert bpy.data.objects["DETECTOR"].data.materials[0] is not None


@step("unregister")
def _():
    neuro3d_lab.unregister()
    assert not hasattr(bpy.types.Scene, "optlab_grid_n")


print("SELFTEST_RESULTS " + json.dumps(results, indent=1))
sys.exit(1 if any(v.startswith("FAIL") for v in results.values()) else 0)
