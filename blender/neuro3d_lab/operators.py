# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators, scene properties and N-panel for Neuro3D Lab."""
import importlib
import os

import bpy
import numpy as np
from bpy.props import FloatProperty, IntProperty, StringProperty

from . import bench_io

DEFAULT_OUT = os.path.join(bench_io.out_root(), "bench")  # $NEURO3D_LAB_OUT or <repo parent>/.cognition/neuro3d-lab


def _out_dir(scene):
    return bpy.path.abspath(scene.optlab_output_dir)


class OPTLAB_OT_build_demo_bench(bpy.types.Operator):
    """Build a demo bench (source, two layers, detector)"""
    bl_idname = "optlab.build_demo_bench"
    bl_label = "Build Demo Bench"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene
        try:
            bench_io.build_demo_bench(s.optlab_grid_n, s.optlab_pixel_m, s.optlab_wavelength_m,
                                      bench_io.DEMO_LAYER_Z_M, bench_io.DEMO_DETECTOR_Z_M, 0)
        except Exception as exc:  # report instead of raising into the UI
            self.report({"ERROR"}, f"Build failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Demo bench created")
        return {"FINISHED"}


class OPTLAB_OT_export_bench(bpy.types.Operator):
    """Export the bench to bench.json and height .npy files"""
    bl_idname = "optlab.export_bench"
    bl_label = "Export Bench"

    def execute(self, context):
        try:
            path = bench_io.export_bench(_out_dir(context.scene))
        except Exception as exc:
            self.report({"ERROR"}, f"Export failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {path}")
        return {"FINISHED"}


class OPTLAB_OT_import_intensity(bpy.types.Operator):
    """Import a detector intensity .npy and show it on the DETECTOR"""
    bl_idname = "optlab.import_intensity"
    bl_label = "Import Intensity"

    def execute(self, context):
        path = bpy.path.abspath(context.scene.optlab_intensity_path)
        if not os.path.isfile(path):
            self.report({"ERROR"}, f"Intensity file not found: {path}")
            return {"CANCELLED"}
        try:
            bench_io.import_intensity(path)
        except Exception as exc:
            self.report({"ERROR"}, f"Import failed: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class OPTLAB_OT_run_bench(bpy.types.Operator):
    """Run optlab.bench.simulate_bench on the exported bench.json"""
    bl_idname = "optlab.run_bench"
    bl_label = "Run Bench"

    def execute(self, context):
        s = context.scene
        src = bpy.path.abspath(s.optlab_src_path)
        bench_io.prefer_optlab_path(src)  # bundled copy first; the UI folder is a fallback
        try:
            bench = importlib.import_module("optlab.bench")
        except ImportError as exc:
            self.report({"ERROR"}, f"optlab not importable (bundled copy or '{src}'): {exc}")
            return {"CANCELLED"}
        json_path = os.path.join(_out_dir(s), "bench.json")
        if not os.path.isfile(json_path):
            self.report({"ERROR"}, "bench.json not found; export the bench first")
            return {"CANCELLED"}
        try:
            result = bench.simulate_bench(json_path)
        except Exception as exc:
            self.report({"ERROR"}, f"simulate_bench failed: {exc}")
            return {"CANCELLED"}
        if isinstance(result, np.ndarray):
            out = os.path.join(_out_dir(s), "detector_intensity.npy")
            np.save(out, result)
            s.optlab_intensity_path = out
        self.report({"INFO"}, "Simulation finished")
        return {"FINISHED"}


class OPTLAB_PT_panel(bpy.types.Panel):
    bl_label = "Neuro3D Lab"
    bl_idname = "OPTLAB_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Neuro3D Lab"

    def draw(self, context):
        col = self.layout.column()
        s = context.scene
        col.prop(s, "optlab_grid_n")
        col.prop(s, "optlab_pixel_m")
        col.prop(s, "optlab_wavelength_m")
        col.operator("optlab.build_demo_bench")
        col.separator()
        col.prop(s, "optlab_output_dir")
        col.operator("optlab.export_bench")
        col.prop(s, "optlab_src_path")
        col.operator("optlab.run_bench")
        col.separator()
        col.prop(s, "optlab_intensity_path")
        col.operator("optlab.import_intensity")


_CLASSES = (OPTLAB_OT_build_demo_bench, OPTLAB_OT_export_bench,
            OPTLAB_OT_import_intensity, OPTLAB_OT_run_bench, OPTLAB_PT_panel)

_PROPS = {
    "optlab_grid_n": IntProperty(name="Grid N", default=bench_io.DEMO_N, min=8, max=4096),
    "optlab_pixel_m": FloatProperty(name="Pixel (m)", default=bench_io.DEMO_PIXEL_M,
                                    min=1e-9, precision=9),
    "optlab_wavelength_m": FloatProperty(name="Wavelength (m)", default=bench_io.DEMO_WAVELENGTH_M,
                                         min=1e-9, precision=9),
    "optlab_output_dir": StringProperty(name="Output dir", default=DEFAULT_OUT, subtype="DIR_PATH"),
    "optlab_intensity_path": StringProperty(name="Intensity .npy", default="", subtype="FILE_PATH"),
    "optlab_src_path": StringProperty(name="optlab src path",
                                      default=bench_io.OPTLAB_SRC, subtype="DIR_PATH"),
}


def register():
    for c in _CLASSES:
        bpy.utils.register_class(c)
    for k, v in _PROPS.items():
        setattr(bpy.types.Scene, k, v)


def unregister():
    for k in _PROPS:
        if hasattr(bpy.types.Scene, k):
            delattr(bpy.types.Scene, k)
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
