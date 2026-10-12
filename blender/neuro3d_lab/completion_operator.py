# SPDX-License-Identifier: GPL-3.0-or-later
"""Versioned Blender operator callback; an absent numerical_config.json uses legacy mode."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import uuid
import bpy
import numpy as np
from . import bench_io


def execute(self, context):
    """Simulate the exported bench and retain numerical settings with its image."""
    scene = context.scene
    root = Path(bpy.path.abspath(scene.optlab_output_dir))
    json_path = root/"bench.json"
    bench_io.prefer_optlab_path(bpy.path.abspath(scene.optlab_src_path))
    if not json_path.is_file():
        self.report({"ERROR"}, "Export the bench before running the simulation")
        return {"CANCELLED"}
    try:
        numerical_path = root/"numerical_config.json"
        if not numerical_path.exists():
            intensity = importlib.import_module("optlab.bench").simulate_bench(json_path)
            out = root/"detector_intensity.npy"
            np.save(out, intensity)
        else:
            settings = json.loads(numerical_path.read_text(encoding="utf-8"))
            if set(settings) != {"padding", "refinement", "boundary", "filter_mode"}:
                raise ValueError("numerical_config.json must contain exactly the four numerical settings")
            adapter = importlib.import_module("neuro3d_completion.bench_adapter")
            result = adapter.simulate_bench_v2(json_path, **settings)
            run = root/"finite_support_runs"/uuid.uuid4().hex
            run.mkdir(parents=True, exist_ok=False)
            out = run/"detector_intensity.npy"
            np.save(out, result["intensity"])
            provenance = dict(settings=settings, bench_sha256=hashlib.sha256(json_path.read_bytes()).hexdigest(),
                              intensity_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                              backend=result["numerical_backend"], scores=result["scores"],
                              scope="relative scalar intensity; no calibrated hardware power")
            (run/"provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        scene.optlab_intensity_path = str(out)
    except Exception as exc:
        self.report({"ERROR"}, f"Simulation failed: {exc}")
        return {"CANCELLED"}
    self.report({"INFO"}, "Simulation finished")
    return {"FINISHED"}
