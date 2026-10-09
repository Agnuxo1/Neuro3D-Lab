import copy
import json
import numpy as np
import pytest
from optlab import simulate_bench, load_bench


def make(tmp_path):
    n = 64
    np.save(tmp_path / "L1.npy", np.zeros((n, n)))
    np.save(tmp_path / "L2.npy", np.random.default_rng(0).uniform(0, 1, (n, n)))
    b = {"schema": "optlab.bench/1", "wavelength_m": 6.33e-7,
         "grid": {"n": n, "pixel_m": 8e-6},
         "source": {"z_m": 0.0},
         "layers": [
             {"name": "L1", "z_m": 0.02, "n_medium": 1.0, "n_material": 1.5, "kappa": 0.0,
              "height_npy": "L1.npy", "height_max_m": 1e-6},
             {"name": "L2", "z_m": 0.03, "n_medium": 1.0, "n_material": 1.5, "kappa": 0.0,
              "height_npy": "L2.npy", "height_max_m": 1e-6}],
         "detector": {"z_m": 0.05, "regions": [
             {"label": 0, "cx_m": 0.0, "cy_m": 0.0, "w_m": 1e-4, "h_m": 1e-4}]},
         "output": {"intensity_npy": "out.npy"}}
    (tmp_path / "bench.json").write_text(json.dumps(b))
    return b


def test_roundtrip(tmp_path):
    b = make(tmp_path)
    I = simulate_bench(tmp_path / "bench.json", output_path=tmp_path / "o.npy")
    assert I.shape == (64, 64) and np.isfinite(I).all()
    assert (tmp_path / "o.npy").exists()
    b0 = copy.deepcopy(b)
    b0["layers"] = []
    I0 = simulate_bench(b0, base_dir=tmp_path)
    assert abs(I.sum() - I0.sum()) / I0.sum() < 1e-6
    assert abs(I0.sum() - 64 * 64) / (64 * 64) < 1e-6
    _, sc = simulate_bench(b, base_dir=tmp_path, return_scores=True)
    assert 0 in sc and np.isfinite(sc[0])
    assert load_bench(tmp_path / "bench.json")["schema"] == "optlab.bench/1"


def test_validation(tmp_path):
    b = make(tmp_path)
    bad = copy.deepcopy(b)
    bad["layers"][1]["z_m"] = 0.01
    with pytest.raises(ValueError):
        simulate_bench(bad, base_dir=tmp_path)
    bad = copy.deepcopy(b)
    bad["detector"]["z_m"] = 0.03
    with pytest.raises(ValueError):
        simulate_bench(bad, base_dir=tmp_path)
    bad = copy.deepcopy(b)
    bad["schema"] = "x"
    with pytest.raises(ValueError):
        simulate_bench(bad, base_dir=tmp_path)
