# SPDX-License-Identifier: GPL-3.0-or-later
"""Headless self-test of the Neuro3D Lab add-on, on the in-domain demo bench (N=256).

Run from anywhere (no absolute paths):
  blender -b --factory-startup -P blender/selftest/run_selftest.py
Output goes to <out>/selftest, where <out> is $NEURO3D_LAB_OUT, or
<repo parent>/.cognition/neuro3d-lab when that variable is unset. Exit code 0 means every step PASS.
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # blender/ folder

import bpy  # noqa: E402
import numpy as np  # noqa: E402

# Expectations of the demo bench. Literal on purpose: they check the constants in bench_io.
N, PIX, WL = 256, 8e-6, 6.33e-7
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

OUT = os.path.join(bench_io.out_root(), "selftest")


@step("register")
def _():
    neuro3d_lab.register()
    assert hasattr(bpy.ops.optlab, "export_bench")
    assert bpy.context.scene.optlab_grid_n == N


@step("build")
def _():
    bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)
    names = {o.name for o in bpy.context.scene.objects}
    assert names == {"SOURCE", "LAYER_1", "LAYER_2", "DETECTOR"}, names


@step("demo_in_domain")
def _():
    # optlab's validity domain: every plane-to-plane spacing must be below z_max
    zmax = bench_io.z_max_m(N, PIX, WL)
    assert abs(zmax - 0.02588) < 1e-4, zmax  # 25.9 mm for N=256
    zs = [0.0, 0.02, 0.04, 0.06]
    gaps = [c - a for a, c in zip(zs, zs[1:])]
    assert max(gaps) < zmax, (gaps, zmax)


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
    assert b["wavelength_m"] == WL, b["wavelength_m"]
    zs = [b["source"]["z_m"]] + [l["z_m"] for l in b["layers"]] + [b["detector"]["z_m"]]
    assert all(a < c for a, c in zip(zs, zs[1:])), zs
    assert max(c - a for a, c in zip(zs, zs[1:])) < bench_io.z_max_m(N, PIX, WL), zs
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
    from neuro3d_lab._vendor.optlab.bench import simulate_bench
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


@step("pack_roundtrip")
def _():
    # hallazgo 1: las alturas deben sobrevivir a guardar y reabrir el .blend
    bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)
    ref = {k: bench_io._image_to_array(bpy.data.images[f"{k}_height"]) for k in ("LAYER_1", "LAYER_2")}
    assert all(a.max() > 0.5 for a in ref.values())
    path = os.path.join(OUT, "pack_test.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    with bpy.data.libraries.load(path) as (src, dst):
        dst.images = [n for n in src.images if n.endswith("_height")]
    assert len(dst.images) == 2, dst.images
    for img in dst.images:
        assert img.packed_file is not None, img.name
        key = "LAYER_1" if img.name.startswith("LAYER_1") else "LAYER_2"
        err = np.abs(bench_io._image_to_array(img) - ref[key]).max()
        assert err < 1e-3, (img.name, err)
        bpy.data.images.remove(img)


@step("trace_scales")
def _():
    # hallazgo 2: escala != 1, z hasta varias BU
    sc0 = bpy.context.scene.unit_settings.scale_length
    try:
        for sc in (1.0, 0.01, 0.001):
            bpy.context.scene.unit_settings.scale_length = sc
            bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)
            hits = bench_io.trace_optical_path((0, 0, 0), (0, 0, 1))
            got = {h["object"]: h["distance_m"] for h in hits}
            for k, z in Z.items():
                assert k in got, (sc, k, hits)
                assert abs(got[k] - z) < 1e-6, (sc, k, got[k], z)
            bpy.context.view_layer.update()
            assert abs(bpy.data.objects["DETECTOR"].location.z * sc - 0.06) < 1e-6
    finally:
        bpy.context.scene.unit_settings.scale_length = sc0
        bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)


@step("build_idempotent")
def _():
    # hallazgo 9: Build dos veces no duplica objetos ni rompe Export
    s = bpy.context.scene
    s.optlab_output_dir = OUT
    for _i in range(2):
        assert bpy.ops.optlab.build_demo_bench() == {"FINISHED"}
    names = sorted(o.name for o in bpy.data.objects)
    assert names == ["DETECTOR", "LAYER_1", "LAYER_2", "SOURCE"], names
    imgs = sorted(i.name for i in bpy.data.images if i.name.startswith("LAYER_"))
    assert imgs == ["LAYER_1_height", "LAYER_2_height"], imgs
    assert bpy.ops.optlab.export_bench() == {"FINISHED"}
    b = json.load(open(os.path.join(OUT, "bench.json")))
    assert [l["name"] for l in b["layers"]] == ["LAYER_1", "LAYER_2"]
    assert max(np.load(os.path.join(OUT, l["height_npy"])).max() for l in b["layers"]) > 0.5


@step("export_clean_numbers")
def _():
    # hallazgo 8: ruido float32 fuera del JSON (camino del operador, propiedades float32)
    s = bpy.context.scene
    s.optlab_wavelength_m = 6.33e-7
    s.optlab_pixel_m = 8e-6
    s.optlab_grid_n = N
    assert bpy.ops.optlab.build_demo_bench() == {"FINISHED"}
    assert bpy.ops.optlab.export_bench() == {"FINISHED"}
    b = json.load(open(os.path.join(OUT, "bench.json")))
    assert b["wavelength_m"] == 6.33e-7, b["wavelength_m"]
    assert b["grid"]["pixel_m"] == 8e-6, b["grid"]["pixel_m"]
    assert [l["z_m"] for l in b["layers"]] == [0.02, 0.04], b["layers"]
    assert b["detector"]["z_m"] == 0.06 and b["source"]["z_m"] == 0.0


@step("orientation_end_to_end")
def _():
    # hallazgo 10: un unico pixel marcado, comprobado contra los datos crudos de Blender
    # (sin usar los helpers de conversion para la comprobacion) y contra detector_masks.
    from neuro3d_lab._vendor.optlab.elements import detector_masks
    bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)
    i0, j0 = 3, 5  # fila desde ARRIBA, columna desde la izquierda
    a = np.zeros((N, N))
    a[i0, j0] = 1.0
    img = bpy.data.images["LAYER_1_height"]
    bench_io._set_image_from_array(img, a)
    raw = np.empty(N * N * 4, dtype=np.float32)
    img.pixels.foreach_get(raw)
    raw = raw.reshape(N, N, 4)[:, :, 0]  # raw[fila desde ABAJO, columna]
    assert raw[N - 1 - i0, j0] == 1.0 and raw.sum() == 1.0, np.argwhere(raw)
    bpy.ops.optlab.export_bench()
    exported = np.load(os.path.join(OUT, "LAYER_1_height.npy"))
    assert tuple(np.argwhere(exported > 0.5)[0]) == (i0, j0) and (exported > 0.5).sum() == 1
    # la geometria del plano coincide: vertice con y>0 <=> v=1 ; x>0 <=> u=1
    obj = bpy.data.objects["LAYER_1"]
    uv = obj.data.uv_layers["UVMap"].data
    for loop in obj.data.loops:
        co = obj.data.vertices[loop.vertex_index].co
        assert (co.x > 0) == (uv[loop.index].uv[0] > 0.5)
        assert (co.y > 0) == (uv[loop.index].uv[1] > 0.5)
    # posicion fisica del texel marcado (u,v) -> metros == celda de detector_masks
    u, v = (j0 + 0.5) / N, (N - 1 - i0 + 0.5) / N
    x_m, y_m = (u - 0.5) * N * PIX, (v - 0.5) * N * PIX
    m = detector_masks(N, PIX, [{"cx_m": x_m, "cy_m": y_m, "w_m": PIX, "h_m": PIX}])[0]
    assert m.sum() == 1 and m[i0, j0] == 1, np.argwhere(m)
    # importacion de intensidad: mismo convenio
    arr = np.zeros((N, N))
    arr[i0, j0] = 1.0
    p = os.path.join(OUT, "marker_intensity.npy")
    np.save(p, arr)
    img2 = bench_io.import_intensity(p)
    raw2 = np.empty(N * N * 4, dtype=np.float32)
    img2.pixels.foreach_get(raw2)
    assert raw2.reshape(N, N, 4)[N - 1 - i0, j0, 0] == 1.0


@step("parented_detector_z")
def _():
    # baja prioridad: z exportado con matrix_world cuando el detector tiene padre.
    # Parent at -5 mm keeps the bench in-domain (detector at 55 mm, 15 mm past LAYER_2).
    bench_io.build_demo_bench(N, PIX, WL, (0.02, 0.04), 0.06, 0)
    par = bpy.data.objects.new("PARENT_EMPTY", None)
    bpy.context.scene.collection.objects.link(par)
    par.location = (0, 0, -0.005)
    det = bpy.data.objects["DETECTOR"]
    det.parent = par
    path = bench_io.export_bench(OUT)
    b = json.load(open(path))
    assert abs(b["detector"]["z_m"] - 0.055) < 1e-9, b["detector"]["z_m"]
    det.parent = None
    bpy.data.objects.remove(par)


@step("unregister")
def _():
    neuro3d_lab.unregister()
    assert not hasattr(bpy.types.Scene, "optlab_grid_n")


print("SELFTEST_RESULTS " + json.dumps(results, indent=1))
n_pass = sum(1 for v in results.values() if v == "PASS")
print(f"SELFTEST_COUNT {n_pass}/{len(results)} PASS")
# Every step must PASS; a PENDING step is not a success (completion guide, G-13).
sys.exit(0 if results and all(v == "PASS" for v in results.values()) else 1)
