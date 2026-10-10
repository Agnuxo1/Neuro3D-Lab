# SPDX-License-Identifier: GPL-3.0-or-later
"""Neuro3D Lab: Blender add-on for virtual optical neural network benches."""

bl_info = {
    "name": "Neuro3D Lab",
    "author": "Neuro3D",
    "version": (0, 1, 1),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Neuro3D Lab",
    "description": "Build, export and visualise optical bench scenes (optlab.bench/1)",
    "category": "Import-Export",
}

from . import bench_io, operators  # noqa: F401


def register():
    operators.register()


def unregister():
    operators.unregister()
