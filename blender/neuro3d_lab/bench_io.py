"""Bench scene building, export (optlab.bench/1), intensity import and ray tracing.

Image orientation convention (fixed, used everywhere):
  numpy array a[i, j]: i = row counted from the TOP of the Blender image,
  j = column counted from the left. Blender stores pixels bottom-up, so
  a = flipud(pixels[:, :, 0].reshape(n, n)). Plane local +x = column
  increasing, local +y = row decreasing (image top = +y).
Units: JSON is in metres; Blender units are converted with
scene.unit_settings.scale_length (metres = BU * scale_length).
"""
import json
import os
import tempfile

import bpy
import numpy as np
from mathutils import Vector

SCHEMA = "optlab.bench/1"
SCALE_PROP = "optlab_intensity_scale"
HEIGHT_IMG_PROP = "optlab_height_image"
EPS_BU = 1e-4

# No absolute paths anywhere: everything is derived from this file's location.
# This file is <repo>/blender/neuro3d_lab/bench_io.py.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPTLAB_SRC = os.path.join(REPO_ROOT, "src")

# In-domain demo bench. optlab's validity domain is plane spacing < z_max = N*pixel^2*n/lambda.
# For N=256, pixel 8 um, lambda 633 nm: z_max = 25.9 mm; the plane spacing is 20 mm.
DEMO_N = 256
DEMO_PIXEL_M = 8e-6
DEMO_WAVELENGTH_M = 6.33e-7
DEMO_LAYER_Z_M = (0.02, 0.04)
DEMO_DETECTOR_Z_M = 0.06


def out_root():
    """Output root: $NEURO3D_LAB_OUT if set, else <system temp dir>/neuro3d-lab."""
    env = os.environ.get("NEURO3D_LAB_OUT", "").strip()
    if env:
        return os.path.abspath(env)
    return os.path.join(tempfile.gettempdir(), "neuro3d-lab")


def z_max_m(n, pixel_m, wavelength_m, medium_index=1.0):
    """Largest plane-to-plane distance (m) inside optlab's validity domain."""
    return n * pixel_m ** 2 * medium_index / wavelength_m


def _scale(scene=None):
    scene = scene or bpy.context.scene
    s = scene.unit_settings.scale_length
    return s if s > 0 else 1.0


def _clean(v):
    """Round to 6 significant digits to drop float32 noise (6.330000132e-7 -> 6.33e-7)."""
    return float(f"{float(v):.6g}")


def _z_world(obj):
    """World-space z of an object (honours parents), in Blender units."""
    return float(obj.matrix_world.translation.z)


def clear_bench():
    """Remove every object created by a previous build (idempotent 'Build')."""
    for obj in [o for o in bpy.data.objects if "optlab_role" in o.keys()]:
        mesh = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for img in [i for i in bpy.data.images
                if i.name.startswith("LAYER_") and i.name.endswith("_height")]:
        bpy.data.images.remove(img)


def _make_plane(name, size_bu, z_bu):
    """Create a square plane mesh object of the given size at height z."""
    h = size_bu / 2.0
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    # uv (0,0) = bottom-left, matching the image bottom-left
    for loop, co in zip(mesh.loops, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[loop.index].uv = co
    obj = bpy.data.objects.new(name, mesh)
    obj.location = (0.0, 0.0, z_bu)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _smooth_pattern(n, rng):
    """Smooth random pattern in [0, 1] (low-pass filtered noise)."""
    noise = rng.standard_normal((n, n))
    f = np.fft.fft2(noise)
    ky = np.fft.fftfreq(n)[:, None]
    kx = np.fft.fftfreq(n)[None, :]
    f *= np.exp(-(kx ** 2 + ky ** 2) / (2 * 0.06 ** 2))
    a = np.real(np.fft.ifft2(f))
    a -= a.min()
    return a / max(a.max(), 1e-12)


def _set_image_from_array(img, arr):
    """Write array a[i, j] (i from top) into a Blender float image (gray RGBA)."""
    n = arr.shape[0]
    flat = np.flipud(arr).astype(np.float32)
    rgba = np.empty((n, n, 4), dtype=np.float32)
    rgba[:, :, 0] = rgba[:, :, 1] = rgba[:, :, 2] = flat
    rgba[:, :, 3] = 1.0
    img.pixels.foreach_set(rgba.ravel())
    img.update()


def _image_to_array(img):
    """Return a[i, j] (i from top) from channel R of a Blender image."""
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return np.flipud(px.reshape(h, w, 4)[:, :, 0]).copy()


def build_demo_bench(n=DEMO_N, pixel_m=DEMO_PIXEL_M, wavelength_m=DEMO_WAVELENGTH_M,
                     layer_z_list=DEMO_LAYER_Z_M, detector_z_m=DEMO_DETECTOR_Z_M, seed=0):
    """Create SOURCE, LAYER_k and DETECTOR planes in the current scene.

    The defaults are the in-domain demo. Plane spacings must stay below
    z_max_m(n, pixel_m, wavelength_m); the self-test checks the demo against it.
    """
    clear_bench()
    scene = bpy.context.scene
    sc = _scale(scene)
    size_bu = n * pixel_m / sc
    rng = np.random.default_rng(seed)

    src = _make_plane("SOURCE", size_bu, 0.0)
    src["optlab_role"] = "source"
    src["optlab_grid_n"] = int(n)
    src["optlab_pixel_m"] = float(pixel_m)
    src["optlab_wavelength_m"] = float(wavelength_m)

    for k, z in enumerate(layer_z_list, start=1):
        obj = _make_plane(f"LAYER_{k}", size_bu, z / sc)
        obj["optlab_role"] = "layer"
        obj["optlab_n_medium"] = 1.0
        obj["optlab_n_material"] = 1.5
        obj["optlab_kappa"] = 0.0
        obj["optlab_height_max_m"] = 1e-6
        name = f"LAYER_{k}_height"
        old = bpy.data.images.get(name)
        if old is not None:
            bpy.data.images.remove(old)
        img = bpy.data.images.new(name, n, n, float_buffer=True, alpha=False)
        img.colorspace_settings.name = "Non-Color"
        _set_image_from_array(img, _smooth_pattern(n, rng))
        img.pack()  # embed pixels so they survive save/reopen of the .blend
        obj[HEIGHT_IMG_PROP] = img

    det = _make_plane("DETECTOR", size_bu, detector_z_m / sc)
    det["optlab_role"] = "detector"
    w = n * pixel_m / 5.0
    c = n * pixel_m / 4.0
    regions = [
        {"label": 0, "cx_m": -c, "cy_m": 0.0, "w_m": w, "h_m": w},
        {"label": 1, "cx_m": 0.0, "cy_m": c, "w_m": w, "h_m": w},
        {"label": 2, "cx_m": c, "cy_m": 0.0, "w_m": w, "h_m": w},
    ]
    det["optlab_regions"] = json.dumps(regions)
    bpy.context.view_layer.update()
    return {"source": src, "detector": det}


def _by_role(role):
    bpy.context.view_layer.update()  # make matrix_world current
    objs = [o for o in bpy.context.scene.objects if o.get("optlab_role") == role]
    return sorted(objs, key=_z_world)


def export_bench(output_dir):
    """Export scene to <output_dir>/bench.json (+ height .npy); return json path."""
    scene = bpy.context.scene
    sc = _scale(scene)
    os.makedirs(output_dir, exist_ok=True)
    srcs, dets = _by_role("source"), _by_role("detector")
    if len(srcs) != 1 or len(dets) != 1:
        raise ValueError("Scene needs exactly one source and one detector")
    src, det = srcs[0], dets[0]
    n = int(src["optlab_grid_n"])
    layers = []
    for obj in _by_role("layer"):
        img = obj.get(HEIGHT_IMG_PROP)
        if img is None or tuple(img.size) != (n, n):
            raise ValueError(f"{obj.name}: missing or wrongly sized {HEIGHT_IMG_PROP}")
        fname = f"{obj.name}_height.npy"
        np.save(os.path.join(output_dir, fname), _image_to_array(img).astype(np.float64))
        layers.append({
            "name": obj.name,
            "z_m": _clean(_z_world(obj) * sc),
            "n_medium": float(obj["optlab_n_medium"]),
            "n_material": float(obj["optlab_n_material"]),
            "kappa": float(obj["optlab_kappa"]),
            "height_npy": fname,
            "height_max_m": float(obj["optlab_height_max_m"]),
        })
    bench = {
        "schema": SCHEMA,
        "wavelength_m": _clean(src["optlab_wavelength_m"]),
        "grid": {"n": n, "pixel_m": _clean(src["optlab_pixel_m"])},
        "source": {"z_m": _clean(_z_world(src) * sc)},
        "layers": layers,
        "detector": {
            "z_m": _clean(_z_world(det) * sc),
            "regions": json.loads(det["optlab_regions"]),
        },
        "output": {"intensity_npy": "detector_intensity.npy"},
    }
    path = os.path.join(output_dir, "bench.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(bench, fh, indent=2)
    return path


def import_intensity(npy_path, name="optlab_detector_intensity"):
    """Load an intensity .npy as image (normalised by max) and show it on DETECTOR."""
    arr = np.load(npy_path).astype(np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected square 2D array, got {arr.shape}")
    peak = float(arr.max())
    norm = arr / peak if peak > 0 else arr
    bpy.context.scene[SCALE_PROP] = peak
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    n = arr.shape[0]
    img = bpy.data.images.new(name, n, n, float_buffer=True, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    _set_image_from_array(img, norm)
    dets = _by_role("detector")
    if dets:
        _assign_emission(dets[0], img, name + "_mat")
    return img


def _assign_emission(obj, img, mat_name):
    """(Re)create an emission material that shows img on obj."""
    mat = bpy.data.materials.get(mat_name)
    if mat is not None:
        bpy.data.materials.remove(mat)
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    emi = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def trace_optical_path(origin_m, direction, max_hits=8):
    """Cast a ray and list successive hits: [{object, distance_m, n_medium}]."""
    scene = bpy.context.scene
    sc = _scale(scene)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    d = Vector(direction).normalized()
    o = Vector(origin_m) / sc
    # Advance in Blender units, well above float32 resolution for |z| up to ~100 BU
    # (ulp ~ 8e-6 at 100 BU); planes closer than EPS_BU in z are not resolved.
    eps = EPS_BU
    start = o.copy()
    hits = []
    for _ in range(max_hits):
        ok, loc, _nrm, _idx, obj, _mat = scene.ray_cast(depsgraph, start, d)
        if not ok:
            break
        hits.append({
            "object": obj.name,
            "distance_m": float((loc - o).dot(d) * sc),
            "n_medium": float(obj.get("optlab_n_medium", 1.0)),
        })
        start = loc + d * eps
    return hits
