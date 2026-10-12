"""Meaningful operator, physics, gradient, evidence, and protocol checks."""
from __future__ import annotations

import gzip
import math
from pathlib import Path
import struct
import numpy as np
import pytest

from neuro3d_completion.core import (Optics, DiffractiveNetwork, detector_masks,
                                      certified_label, score_error_bound)
from neuro3d_completion.data import encode, read_idx
from neuro3d_completion.evidence import (holm, paired_comparison, crossed_bootstrap,
                                         reserve_test, write_json_new, contraction_budget)
from neuro3d_completion.interfaces import oblique, normal_incidence
from neuro3d_completion.hardware import compare


def masks(n=16):
    result = np.zeros((3, n, n))
    for k in range(3):
        result[k, 5:8, 2+4*k:5+4*k] = 1
    return result


def fields(n=16):
    rng = np.random.default_rng(15)
    value = rng.normal(size=(2, n, n))+1j*rng.normal(size=(2, n, n))
    return value/np.linalg.norm(value, axis=(1, 2))[:, None, None]


@pytest.mark.parametrize("boundary,padding", [("periodic", 1), ("transparent", 2), ("aperture", 2)])
@pytest.mark.parametrize("filter_mode", ["none", "phase_gradient"])
def test_adjoint_and_passivity(boundary, padding, filter_mode):
    model = DiffractiveNetwork(Optics(n=16, padding=padding, boundary=boundary, filter_mode=filter_mode), masks())
    rng = np.random.default_rng(2)
    a = rng.normal(size=(model.optics.m, model.optics.m))+1j*rng.normal(size=(model.optics.m, model.optics.m))
    b = rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
    pa = model.propagate(a, model.kernels[0])
    phb = model.propagate(b, model.kernels[0], adjoint=True)
    assert np.allclose(np.vdot(pa, b), np.vdot(a, phb), rtol=1e-12, atol=1e-12)
    assert np.linalg.norm(pa) <= np.linalg.norm(a)*(1+1e-12)


@pytest.mark.parametrize("boundary,padding", [("periodic", 1), ("transparent", 2), ("aperture", 2)])
@pytest.mark.parametrize("refinement", [1, 2])
def test_gradient_independent_entries(boundary, padding, refinement):
    model = DiffractiveNetwork(Optics(n=16, padding=padding, boundary=boundary, refinement=refinement), masks(), beta=10)
    u, labels = fields(), np.array([0, 2])
    _, grad = model.loss_and_grad(u, labels)
    rng = np.random.default_rng(11)
    for flat in rng.choice(grad.size, 12, replace=False):
        index = np.unravel_index(flat, grad.shape)
        original, h = model.phases[index], 1e-5
        model.phases[index] = original+h; plus = model.loss_and_grad(u, labels)[0]
        model.phases[index] = original-h; minus = model.loss_and_grad(u, labels)[0]
        model.phases[index] = original
        assert np.isclose((plus-minus)/(2*h), grad[index], atol=2e-9, rtol=1e-5)


def test_fixed_parameter_count_and_global_phase():
    a = DiffractiveNetwork(Optics(n=16, padding=2), masks())
    b = DiffractiveNetwork(Optics(n=16, padding=4), masks())
    assert a.phases.size == b.phases.size == 3*16**2
    x = fields()
    before = a.forward(x)
    a.phases += .37
    after = a.forward(x)
    assert np.allclose(before["scores"], after["scores"], rtol=1e-12, atol=1e-12)
    assert np.allclose(before["I"], after["I"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("parameters", [{"n":15}, {"padding":0}, {"pixel_m":0},
                                       {"wavelength_m":float("nan")}, {"distances_m":(-1,)},
                                       {"boundary":"periodic", "padding":2}])
def test_invalid_optics(parameters):
    with pytest.raises(ValueError):
        Optics(**parameters)


def test_invalid_labels_and_batches():
    model = DiffractiveNetwork(Optics(n=16), masks())
    for labels in ([0, 4], [0], [-1, 0], [0.5, 1]):
        with pytest.raises(ValueError):
            model.loss_and_grad(fields(), labels)
    with pytest.raises(ValueError):
        model.fit(fields(), [0], epochs=1)
    with pytest.raises(ValueError):
        model.forward(np.full((16, 16), np.nan))


def test_error_and_margin_certificate():
    model = DiffractiveNetwork(Optics(n=16), masks())
    f = model.forward(fields()[0])["f"]
    rng = np.random.default_rng(9)
    e = rng.normal(size=f.shape)+1j*rng.normal(size=f.shape)
    e *= 1e-3/np.linalg.norm(e)
    actual = np.abs(np.einsum("kij,ij->k", model.full_masks, np.abs(f+e)**2-np.abs(f)**2))
    bound = score_error_bound(f, np.linalg.norm(e), model.full_masks)
    assert actual.max() <= bound
    assert certified_label([.4, .8], .1)["certified"]
    assert not certified_label([.4, .6], .1)["certified"]
    assert np.isclose(contraction_budget(1, .01, [.02, .03], [.04]), .10)


def test_paired_statistics_and_shared_test():
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])
    y = np.arange(20)%2
    a, b = y.copy(), y.copy(); b[:10] = 1-b[:10]
    result = paired_comparison(y, a, b)
    assert result["first_only"] == 10 and result["second_only"] == 0
    assert result["p_two_sided"] < .01
    same = np.stack([a, a])
    assert crossed_bootstrap(y, same, same, repetitions=100)["percentile_95"] == [0, 0]


def test_reservation_rejects_configuration_change(tmp_path):
    test = tmp_path/"test.dat"; test.write_bytes(b"fixed data")
    cfg1, cfg2 = tmp_path/"c1.json", tmp_path/"c2.json"
    write_json_new(cfg1, {"lr":.01}); write_json_new(cfg2, {"lr":.02})
    reserve_test(tmp_path/"ledger", "study", "d2nn", 0, test, cfg1)
    with pytest.raises(FileExistsError):
        reserve_test(tmp_path/"ledger", "study", "d2nn", 0, test, cfg2)


def test_fresnel_brewster_tir_and_validation():
    assert np.isclose(normal_incidence(1, 1.5)["R"], .04)
    assert oblique(1, 1.5, math.atan(1.5), "p")["R"] < 1e-28
    for pol in ("s", "p"):
        result = oblique(1.5, 1, math.pi/3, pol)
        assert np.isclose(result["R"], 1) and result["T"] == 0 and abs(result["t"]) > 0
    with pytest.raises(ValueError):
        oblique(1, 1.5, math.pi/2, "s")


def test_idx_and_energy_encoding(tmp_path):
    path = tmp_path/"labels.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 3)+bytes([0, 1, 2]))
    assert read_idx(path).tolist() == [0, 1, 2]
    x = encode(np.ones((2, 8, 8)))
    assert x.shape == (2, 64, 64) and np.allclose((x*x).sum(axis=(1, 2)), 1)
    with pytest.raises(ValueError):
        encode(np.zeros((1, 8, 8)))


def test_hardware_comparison_with_known_calibration(tmp_path):
    sim, measured = tmp_path/"sim.npz", tmp_path/"measured.npz"
    predicted = np.array([[.8, .1], [.1, .7]])
    np.savez(sim, region_energy=predicted, sample_ids=[0, 1])
    powers = np.stack([predicted*2, predicted*2])
    np.savez(measured, power_w=powers, dark_w=np.zeros_like(powers), input_power_w=[2, 2],
             labels=[0, 1], sample_ids=[0, 1], unit="W")
    result = compare(measured, sim)
    assert result["relative_L2"] == 0 and result["accuracy"] == 1


@pytest.mark.parametrize("boundary,padding", [("periodic", 1), ("transparent", 2), ("aperture", 2)])
@pytest.mark.parametrize("refinement", [1, 2])
def test_independent_torch_gradient(boundary, padding, refinement):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from neuro3d_completion.torch_backend import check_equivalence
    model = DiffractiveNetwork(Optics(n=16, boundary=boundary, padding=padding, refinement=refinement), masks())
    assert check_equivalence(model, fields(), np.array([0, 2]))["pass"]


def test_parameter_only_restore(tmp_path):
    torch = pytest.importorskip("torch")
    from neuro3d_completion.benchmark import restore_parameters
    from neuro3d_completion.torch_backend import TorchD2NN
    a = TorchD2NN(DiffractiveNetwork(Optics(n=16, refinement=2), masks(), seed=0))
    b = TorchD2NN(DiffractiveNetwork(Optics(n=16, refinement=2), masks(), seed=1))
    path = tmp_path/"weights.pt"
    torch.save({key: value.detach().clone() for key, value in a.named_parameters()}, path)
    restore_parameters(b, path)
    u = torch.tensor(fields(), dtype=torch.complex128)
    assert torch.allclose(a(u), b(u), rtol=1e-12, atol=1e-12)


def test_quadratic_readout_structure():
    torch = pytest.importorskip("torch")
    from neuro3d_completion.torch_backend import digital_model
    model = digital_model("quadratic12")
    x = torch.randn(2, 1, 64, 64, dtype=torch.float64)
    scores = model(x)
    assert sum(p.numel() for p in model.parameters()) == 49272
    assert torch.all(scores >= 0)
    assert torch.allclose(model(2*x), 4*scores, rtol=1e-12, atol=1e-12)
