# SPDX-License-Identifier: GPL-3.0-or-later
"""Check the add-on installed from its built zip in a clean Blender profile, without the repository.

Install first into an empty profile, then run this script with the same profile:
  set BLENDER_USER_RESOURCES=<empty folder>
  blender -b --factory-startup --command extension install-file -r user_default <neuro3d_lab-X.zip>
  blender -b --factory-startup -P check_installed_extension.py
Environment: NEURO3D_CHECK_OUT (output folder), NEURO3D_CHECK_REFERENCE (optional legacy intensity .npy
produced by the repository self-test, compared to 1e-12). Prints INSTALLED_EXTENSION_OK on success.
"""
import json
import os
import sys

import addon_utils
import bpy
import numpy as np

OUT = os.environ["NEURO3D_CHECK_OUT"]
REFERENCE = os.environ.get("NEURO3D_CHECK_REFERENCE")
MODULE = "bl_ext.user_default.neuro3d_lab"

bpy.ops.wm.read_factory_settings(use_empty=True)
assert not any("Neuro3D-Lab" in p for p in sys.path), "the repository must not be on sys.path"
addon_utils.enable(MODULE, default_set=True)
package = sys.modules[MODULE]
installed = os.path.dirname(os.path.abspath(package.__file__))
print("INSTALLED_AT", installed)
assert "Neuro3D-Lab" not in installed, installed

os.makedirs(OUT, exist_ok=False)
scene = bpy.context.scene
scene.optlab_output_dir = OUT
scene.optlab_src_path = os.path.join(OUT, "no_such_src")
assert bpy.ops.optlab.build_demo_bench() == {"FINISHED"}
assert bpy.ops.optlab.export_bench() == {"FINISHED"}
assert bpy.ops.optlab.run_bench() == {"FINISHED"}
legacy = np.load(os.path.join(OUT, "detector_intensity.npy"))
import optlab  # noqa: E402
import neuro3d_completion  # noqa: E402
for module in (optlab, neuro3d_completion):
    origin = os.path.abspath(module.__file__)
    print("MODULE_ORIGIN", module.__name__, origin)
    assert origin.startswith(os.path.join(installed, "_vendor")), origin
if REFERENCE:
    reference = np.load(REFERENCE)
    print("MAXDIFF_VS_REPOSITORY_RUN", float(np.abs(legacy - reference).max()))
    assert np.allclose(legacy, reference, rtol=0, atol=1e-12)
with open(os.path.join(OUT, "numerical_config.json"), "x", encoding="utf-8") as f:
    json.dump({"padding": 2, "refinement": 1, "boundary": "aperture", "filter_mode": "none"}, f)
assert bpy.ops.optlab.run_bench() == {"FINISHED"}
provenance = json.load(open(os.path.join(os.path.dirname(scene.optlab_intensity_path), "provenance.json")))
assert provenance["backend"] == "neuro3d_completion/v1", provenance
print("FINITE_SUPPORT_SCORES", json.dumps(provenance["scores"]))
addon_utils.disable(MODULE, default_set=True)
print("INSTALLED_EXTENSION_OK")
