"""Render an orbit of the in-domain demo bench (N=256) with the optlab intensity on the
DETECTOR, then assemble the frames into docs/img/blender_bench.gif.

Two steps, both run from the repository (no absolute paths):
  1. blender -b --factory-startup -P blender/render_bench_gif.py
       renders the PNG frames (Blender's bundled Python has no Pillow).
  2. python blender/render_bench_gif.py --gif
       assembles the frames into docs/img/blender_bench.gif with Pillow.
Frames go to <out>/render/frames, where <out> is $NEURO3D_LAB_OUT, or the system temp
dir plus neuro3d-lab when that variable is unset.

Blender only visualises; the physics (simulate_bench) is computed by optlab.

Scale, stated honestly: the bench is physical in z (planes 0, 20, 40 and 60 mm, so the
spacing is 20 mm) and the pixel is physical (8 um). The planes are NOT drawn to scale in
x and y: each plane is magnified VIS_XY_SCALE = 15 times in x and y for visualization,
so a 2.048 mm plane is drawn about 31 mm wide, the same width as in the earlier GIF.
The image is a schematic, not a to-scale picture of the optics.
"""
import glob
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_env_out = os.environ.get("NEURO3D_LAB_OUT", "").strip()
OUT_ROOT = os.path.abspath(_env_out) if _env_out else os.path.join(tempfile.gettempdir(), "neuro3d-lab")
FRAMES = os.path.join(OUT_ROOT, "render", "frames")
GIF = os.path.join(ROOT, "docs", "img", "blender_bench.gif")
NFRAMES = 36
VIS_XY_SCALE = 15.0          # visual enlargement of the physical planes (x, y only)
GIF_MS_PER_FRAME = 90
GIF_MAX_BYTES = 2 * 1024 * 1024


def assemble_gif():
    """Frames -> GIF with Pillow. Run with a plain Python that has Pillow installed."""
    from PIL import Image

    files = sorted(glob.glob(os.path.join(FRAMES, "f[0-9][0-9][0-9].png")))
    if len(files) < 20:
        raise SystemExit(f"need at least 20 frames in {FRAMES}, found {len(files)}")
    # one shared palette (from a sample of frames) so the colours do not flicker between frames
    picks = [Image.open(p).convert("RGB") for p in files[:: max(1, len(files) // 6)]]
    w, h = picks[0].size
    sheet = Image.new("RGB", (w, h * len(picks)))
    for i, im in enumerate(picks):
        sheet.paste(im, (0, i * h))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = []
    for p in files:
        with Image.open(p) as im:
            frames.append(im.convert("RGB").quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG))
    os.makedirs(os.path.dirname(GIF), exist_ok=True)
    frames[0].save(GIF, save_all=True, append_images=frames[1:], duration=GIF_MS_PER_FRAME,
                   loop=0, optimize=True)
    size = os.path.getsize(GIF)
    print(f"GIF {GIF}: {len(frames)} frames, {size} bytes")
    if size > GIF_MAX_BYTES:
        raise SystemExit(f"GIF is {size} bytes, above the 2 MB limit")


if "--gif" in sys.argv:
    assemble_gif()
    sys.exit(0)

# ---- Blender side: render the frames -------------------------------------------------
import bpy  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from neuro3d_lab import bench_io  # noqa: E402
from optlab.bench import simulate_bench  # noqa: E402

N, PIX, WL = bench_io.DEMO_N, bench_io.DEMO_PIXEL_M, bench_io.DEMO_WAVELENGTH_M
Z_LAYERS, Z_DET = bench_io.DEMO_LAYER_Z_M, bench_io.DEMO_DETECTOR_Z_M
RENDER = os.path.dirname(FRAMES)
os.makedirs(FRAMES, exist_ok=True)
for _old in glob.glob(os.path.join(FRAMES, "f*.png")):
    os.remove(_old)

bpy.ops.wm.read_factory_settings(use_empty=True)
bench_io.build_demo_bench(N, PIX, WL, Z_LAYERS, Z_DET, 0)
bench_io.export_bench(RENDER)
inten = np.asarray(simulate_bench(os.path.join(RENDER, "bench.json")))
assert inten.shape == (N, N) and np.isfinite(inten).all() and inten.max() > 0
np.save(os.path.join(RENDER, "detector_intensity.npy"), inten)
print("intensity", inten.shape, float(inten.min()), float(inten.max()))
bench_io.import_intensity(os.path.join(RENDER, "detector_intensity.npy"))

objs = bpy.data.objects
for o in objs:
    o.scale = (VIS_XY_SCALE, VIS_XY_SCALE, 1.0)


def mat(name, build):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    build(nt, out)
    return m


# detector: boost emission of the imported material
dm = objs["DETECTOR"].data.materials[0]
for nd in dm.node_tree.nodes:
    if nd.type == "EMISSION":
        nd.inputs["Strength"].default_value = 3.0

# layers: semi-transparent tinted emission modulated by the height image
for k, tint in ((1, (0.2, 0.55, 1.0, 1)), (2, (0.7, 0.35, 1.0, 1))):
    o = objs[f"LAYER_{k}"]
    img = o[bench_io.HEIGHT_IMG_PROP]

    def build(nt, out, img=img, tint=tint):
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
        tex.interpolation = "Linear"
        mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
        mul.inputs[0].default_value = 1.0
        mul.inputs[6].default_value = tint
        nt.links.new(tex.outputs["Color"], mul.inputs[7])
        emi = nt.nodes.new("ShaderNodeEmission"); emi.inputs["Strength"].default_value = 1.6
        nt.links.new(mul.outputs[2], emi.inputs["Color"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader"); mix.inputs[0].default_value = 0.45
        nt.links.new(tr.outputs[0], mix.inputs[1]); nt.links.new(emi.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    o.data.materials.clear(); o.data.materials.append(mat(f"L{k}_mat", build))


def src_build(nt, out):
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Color"].default_value = (1.0, 0.25, 0.2, 1); emi.inputs["Strength"].default_value = 2.0
    nt.links.new(emi.outputs[0], out.inputs["Surface"])
objs["SOURCE"].data.materials.clear(); objs["SOURCE"].data.materials.append(mat("SRC_mat", src_build))

# optical axis line (not magnified: it is a 0.5 mm guide, drawn on the z axis)
cu = bpy.data.curves.new("axis", "CURVE"); cu.dimensions = "3D"
sp = cu.splines.new("POLY"); sp.points.add(1)
sp.points[0].co = (0, 0, 0, 1); sp.points[1].co = (0, 0, Z_DET, 1)
cu.bevel_depth = 0.00025
axis = bpy.data.objects.new("AXIS", cu); bpy.context.scene.collection.objects.link(axis)
axis.data.materials.append(mat("AXIS_mat", lambda nt, out: (
    lambda e: (e.inputs["Color"].__setattr__("default_value", (1, 0.8, 0.3, 1)),
               e.inputs["Strength"].__setattr__("default_value", 2.0),
               nt.links.new(e.outputs[0], out.inputs["Surface"])))(nt.nodes.new("ShaderNodeEmission"))))

scene = bpy.context.scene
w = bpy.data.worlds.new("W"); scene.world = w; w.use_nodes = True
bg = w.node_tree.nodes["Background"]
c = [int("0d1117"[i:i + 2], 16) / 255 for i in (0, 2, 4)]
lin = [((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c]
bg.inputs["Color"].default_value = (*lin, 1); bg.inputs["Strength"].default_value = 1.0

scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 24
scene.cycles.use_denoising = False
scene.cycles.transparent_max_bounces = 8
scene.render.threads_mode = "FIXED"; scene.render.threads = 2
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 640, 360, 100
scene.view_settings.view_transform = "Standard"
scene.render.image_settings.file_format = "PNG"

target = bpy.data.objects.new("T", None); scene.collection.objects.link(target)
target.location = (0, 0, 0.03)
cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); scene.collection.objects.link(cam)
cam.data.lens = 35
tc = cam.constraints.new("TRACK_TO"); tc.target = target
tc.track_axis = "TRACK_NEGATIVE_Z"; tc.up_axis = "UP_Y"
scene.camera = cam

R, EL0 = 0.17, math.radians(28)
for f in range(NFRAMES):
    a = math.radians(-40) + 2 * math.pi * f / NFRAMES * 0.5   # half turn of the orbit
    el = EL0 + math.radians(6) * math.sin(2 * math.pi * f / NFRAMES)
    cam.location = (R * math.cos(el) * math.sin(a), -R * math.cos(el) * math.cos(a), 0.03 + R * math.sin(el))
    bpy.context.view_layer.update()
    scene.render.filepath = os.path.join(FRAMES, f"f{f:03d}.png")
    bpy.ops.render.render(write_still=True)
    print("frame", f, flush=True)
print("RENDER_DONE frames in", FRAMES, "- next: python blender/render_bench_gif.py --gif")
