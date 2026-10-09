import copy
import json
import numpy as np
import pytest
from optlab import simulate_bench, load_bench
from optlab.propagation import angular_spectrum


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
    # las capas con altura aleatoria deben cambiar el resultado (hallazgo 7)
    assert np.linalg.norm(I - I0) / np.linalg.norm(I0) > 1e-3
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


def _structured_source(tmp_path, n=64):
    rng = np.random.default_rng(5)
    np.save(tmp_path / "src.npy", rng.uniform(0.0, 1.0, (n, n)))


def test_medium_index_changes_propagation(tmp_path):
    # hallazgo 3: n_medium = n_material = 1.5 (sin fase) debe propagar con lambda/1.5
    n = 64
    _structured_source(tmp_path, n)
    np.save(tmp_path / "L1.npy", np.zeros((n, n)))
    base = {"schema": "optlab.bench/1", "wavelength_m": 6.33e-7,
            "grid": {"n": n, "pixel_m": 8e-6},
            "source": {"z_m": 0.0, "amplitude_npy": "src.npy"},
            "layers": [{"name": "L1", "z_m": 0.01, "n_medium": 1.0, "n_material": 1.0,
                        "kappa": 0.0, "height_npy": "L1.npy", "height_max_m": 1e-6}],
            "detector": {"z_m": 0.02, "regions": []}}
    I_vac = simulate_bench(base, base_dir=tmp_path)
    med = copy.deepcopy(base)
    med["medium_index"] = 1.5
    med["layers"][0]["n_medium"] = 1.5
    med["layers"][0]["n_material"] = 1.5
    I_med = simulate_bench(med, base_dir=tmp_path)
    assert np.linalg.norm(I_med - I_vac) / np.linalg.norm(I_vac) > 1e-3
    u0 = np.load(tmp_path / "src.npy").astype(complex)
    ref = np.abs(angular_spectrum(u0, 8e-6, 6.33e-7 / 1.5, 0.02)) ** 2
    assert np.abs(I_med - ref).max() < 1e-9 * ref.max()


def test_medium_index_mismatch_raises(tmp_path):
    b = make(tmp_path)
    b["medium_index"] = 1.5  # capas siguen con n_medium = 1.0
    with pytest.raises(ValueError):
        simulate_bench(b, base_dir=tmp_path)
    b["medium_index"] = 1.0
    simulate_bench(b, base_dir=tmp_path)


def test_input_validation_extras(tmp_path):
    b = make(tmp_path)
    for mut in (lambda d: d["layers"][0].__setitem__("height_max_m", -1e-6),
                lambda d: d["detector"]["regions"][0].__setitem__("w_m", 0.0),
                lambda d: d["detector"]["regions"][0].__setitem__("h_m", -1e-4)):
        bad = copy.deepcopy(b)
        mut(bad)
        with pytest.raises(ValueError):
            simulate_bench(bad, base_dir=tmp_path)
    h = np.zeros((64, 64))
    h[3, 3] = np.nan
    np.save(tmp_path / "L1.npy", h)
    with pytest.raises(ValueError):
        simulate_bench(b, base_dir=tmp_path)


def test_evanescent_components_dropped(tmp_path):
    # tablero de ajedrez con pixel << lambda/2: solo ondas evanescentes -> ~0 energia
    n = 32
    ii, jj = np.indices((n, n))
    np.save(tmp_path / "chk.npy", ((ii + jj) % 2).astype(float))
    b = {"schema": "optlab.bench/1", "wavelength_m": 6.33e-7,
         "grid": {"n": n, "pixel_m": 1e-7},
         "source": {"z_m": 0.0, "amplitude_npy": "chk.npy"},
         "layers": [], "detector": {"z_m": 1e-5, "regions": []}}
    I = simulate_bench(b, base_dir=tmp_path)
    mean_free = (0.5 ** 2) * n * n  # energia del termino de frecuencia cero propagante
    assert I.sum() < 1.05 * mean_free  # el patron alterno (evanescente) no aporta
    assert I.std() < 1e-6 * I.mean()  # solo queda la componente uniforme
