# Blender Extensions review response — 2026-10-13

Review: https://extensions.blender.org/approval-queue/neuro3d-lab/#activity-23142

Nickberckley requested removal of manual sys-path changes and an English website description. Version 0.2.1 addresses both requests. The old package remains in the version history.

## Implementation

The extension uses relative imports under `bl_ext.<repository>.neuro3d_lab._vendor`; it imports neither numerical library by a global name. The external source-folder UI and path-selection helper were removed. The bundled `optlab` files remain identical to source. The bundled completion adapter has exactly one packaging adaptation (`from ..optlab.bench import load_bench`); all other content remains identical to source and is checked by the vendor test. No file in `src/` or any scientific contract was changed.

Output defaults to Blender user DATAFILES, rather than a location derived from the extension installation directory. The environment override and user-selected output folder remain supported.

## Validation

- Repository tests: 100 passed in 86 seconds (Python 3.13, CPU). `PYTHONPATH` was set to this checkout's `src` only for the repository test process so subprocess examples could import the uninstalled source; it is absent from the Blender profile check. An earlier unconfigured run had 96 passes and four import errors in subprocess examples.
- Blender 4.5.14 LTS: legacy self-test 15/15, completion self-test 8/8.
- Official extension manifest validation, ZIP build and clean-profile install/check: exit code 0.
- Installed legacy intensity vs repository run: maximum absolute difference 0.0. Periodic completion/legacy relative difference: 7.822519740818624e-16.
- Installed package works with sentinel modules occupying both global library names, leaves `sys.path` unchanged during simulation and unregister, and imports its result into Blender.
- ZIP: 24 entries, no bytecode or self-test files. SHA-256: `f5937c54315ea1e1aba42acc0dd244679da00296e99bd3173aa49a412b6910d6`.

Logs and summary: `evidence/blender-review-20261013/`. Upload hash shown by Blender Extensions matches the local ZIP. Version URL: https://extensions.blender.org/add-ons/neuro3d-lab/versions/#v021 . Website description is English with a Spanish summary; release notes are English. Approval is pending.
