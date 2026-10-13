"""The executable examples run (guide G-26); example 3 is checked for its TEST refusal."""
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    done = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_example_scalar_reference():
    assert "ASM_vs_RS_complex_relative_L2" in run("examples/01_scalar_reference.py", "--n", "16")


def test_example_convergence():
    assert "observed order" in run("examples/02_convergence_study.py")


def test_example_hardware_synthetic(tmp_path):
    assert "SYNTHETIC DATA" in run("examples/04_hardware_analysis_synthetic.py", "--output", str(tmp_path/"hw"))


def test_example_inference_refuses_test(tmp_path):
    np.savez(tmp_path/"test.npz", images=np.zeros((1, 28, 28)), labels=[0], sample_ids=[0])
    done = subprocess.run([sys.executable, "examples/03_inference_saved_masks.py", "--phases",
                           "experiments/results/d2nn_seed0_phases.npz", "--split", str(tmp_path/"test.npz")],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode != 0 and "reserved split" in done.stderr
