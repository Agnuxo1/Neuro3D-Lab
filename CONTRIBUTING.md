# Contributing to Neuro3D-Lab

Neuro3D-Lab has two parts:

- **optlab**, the numerical core in `src/optlab` (scalar-field propagation, Fresnel diffraction, interferometers, diffractive neural networks).
- **Neuro3D Lab**, the Blender add-on in `blender/neuro3d_lab`, which builds the bench and displays the result.

## Setup

Python 3.10 or newer is required. CI uses Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

Optional extras:

- `experiments`: scikit-learn, for the digit-classification experiment.
- `figures`: matplotlib, pillow and scikit-learn, for regenerating the figures.

## Checks

- `pytest -q` must pass before you open a pull request. CI runs the same command on Python 3.13 (ubuntu-latest) for every push and pull request to `master`.
- Blender is not run in CI. To exercise the add-on, use a local Blender 4.5 binary: `blender -b --factory-startup -P blender/selftest/run_selftest.py`. Attach the log to the pull request when the add-on changes.
- Validate the add-on manifest: `blender -b --factory-startup --command extension validate blender/neuro3d_lab`.

## Output locations

Do not commit generated data. Large or raw outputs go to a folder outside the repository, chosen with the `NEURO3D_LAB_OUT` environment variable (default: `<system temp>/neuro3d-lab`). Only small summaries belong in the repository, for example `experiments/results/validation_summary.json`.

Do not write absolute paths into code, tests or documentation. Build paths from `pathlib` and the repository root, or read them from environment variables.

## Style

- Follow PEP 8 and the style of the surrounding code.
- Keep network access out of `src/optlab`. File writes go only to paths the caller supplies; do not write to fixed or hard-coded locations.
- Keep runtime dependencies to numpy and scipy. Any new dependency goes into an optional extra and needs a justification in the pull request.
- State units in variable names or docstrings. Internal computations use SI units (metres, seconds).

## Proposing changes

1. For anything that changes results, physical models, the `optlab.bench/1` file format or a public API, open an issue first and agree on the approach.
2. Fork the repository, branch from `master`, and keep each pull request to one concern.
3. Add or update tests for the behaviour you change.
4. In the pull request, describe what changed, list the commands you ran and their results, and state whether any contract in `docs/CONTRACTS/` is affected.

## Contract discipline

Measurable claims are governed by preregistered contracts in `docs/CONTRACTS/`. Each contract fixes its thresholds and its procedure before any measurement is taken.

- Do not change a threshold after seeing a result. If a threshold turns out to be wrong, propose a new contract version and say so in the pull request.
- Report failures. A criterion that fails stays in the record with its numbers. It is not removed or rewritten.
- Exploratory experiments are labelled as exploratory and carry no verdict.

## Licence

Contributions to the core library `src/optlab/` are accepted under the Apache License, Version 2.0 ([LICENSE](LICENSE)). Contributions to the Blender add-on in `blender/` are accepted under the GNU General Public License, version 3 or later ([blender/neuro3d_lab/LICENSE](blender/neuro3d_lab/LICENSE)).

## Conduct and security

- Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). Do not open a public issue for them.
