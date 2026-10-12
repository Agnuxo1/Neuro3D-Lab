"""Checks for the research-excellence additions: references, convergence, statistics,
noise, robustness, expressivity, data provenance, claims, blinding and closure."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import pytest

from neuro3d_completion.core import Optics, DiffractiveNetwork
from neuro3d_completion import (blind, budget, claims, convergence, expressivity, photon_budget, pixel_reference,
                                power, qmnist, robustness, rs_network)
from neuro3d_completion.evidence import digest_file, paired_comparison
from neuro3d_completion.validation import relative

ROOT = Path(__file__).resolve().parents[1]


def small_masks(n=8):
    masks = np.zeros((2, n, n))
    masks[0, 2:4, 2:4] = 1
    masks[1, 4:6, 4:6] = 1
    return masks


def smooth_inputs(n=8, count=3, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros((count, n, n))
    x[:, n//4:3*n//4, n//4:3*n//4] = rng.uniform(0.2, 1, (count, n//2, n//2))
    return x/np.linalg.norm(x, axis=(1, 2))[:, None, None]


def test_pixel_reference_smooth_screen_agrees_with_rs():
    result = pixel_reference.pixel_reference(n=8, refinements=(1, 3), paddings=(8,), quadratures=(1, 3, 9),
                                             phases=np.zeros((8, 8)), filter_modes=("none",))
    assert result["rs_quadrature"]["q3_vs_q9"] < result["rs_quadrature"]["q1_vs_q3"]
    assert result["asm"]["none_r3_p8"]["field_vs_rs_finest"] < 5e-3


def test_pixel_reference_random_screen_quadrature_converges():
    result = pixel_reference.pixel_reference(n=8, refinements=(1,), paddings=(4,), quadratures=(1, 3, 9),
                                             filter_modes=("none",))
    q = result["rs_quadrature"]
    assert q["q3_vs_q9"] < q["q1_vs_q3"]/5


def test_rs_network_matches_core_on_large_window():
    n, a, lam = 8, 4e-4, 7.5e-4
    phases = np.random.default_rng(3).uniform(0, 2*np.pi, (1, n, n))
    u = smooth_inputs(n, 1)[0]
    field = rs_network.rs_network_fields(u, phases, a, lam, (0.006, 0.006), q=3)[0]
    errors = {}
    for padding in (16, 64):
        model = DiffractiveNetwork(Optics(n=n, pixel_m=a, wavelength_m=lam, distances_m=(0.006, 0.006),
                                          padding=padding, refinement=3, boundary="aperture"), small_masks(n))
        model.phases = phases.copy()
        errors[padding] = relative(model.forward(u)["f"][model.roi]*3/a, field)
    # The window error shrinks with padding; the residual near 3e-3 at d = 8 wavelengths is the
    # evanescent field, which RS includes and the core propagator deliberately omits.
    assert errors[64] < errors[16] and errors[64] < 5e-3
    scores = rs_network.rs_scores(field[None], small_masks(n), a, 3, beta=model.beta)[0]
    assert np.allclose(scores, model.forward(u)["scores"], rtol=2e-2)


def test_gci_recovers_second_order_and_flags_oscillation():
    h = np.array([1.0, 0.5, 0.25])
    phi = 3.0 + 0.7*h**2
    result = convergence.gci_three_level(phi[2], phi[1], phi[0], ratio=2)
    assert result["status"] == "monotone"
    assert np.isclose(result["order"], 2) and np.isclose(result["extrapolated"], 3.0)
    oscillating = convergence.gci_three_level(1.0, 1.1, 0.95, ratio=2)
    assert oscillating["status"] == "oscillatory" and np.isnan(oscillating["gci_fine"])
    d = convergence.observed_order_from_differences(0.28, 0.06, 2)
    assert d["status"] == "monotone" and 2.1 < d["order"] < 2.3
    assert convergence.observed_order_from_differences(0.1, 0.2, 2)["status"] == "not converging"


def test_gci_from_record(tmp_path):
    path = tmp_path/"pixel.json"
    path.write_text(json.dumps({"values": {
        "refinement_1_vs_2": {"field_relative_L2": 0.4}, "refinement_2_vs_4": {"field_relative_L2": 0.1},
        "refinement_4_vs_8": {"field_relative_L2": 0.025}}}))
    report = convergence.from_record(path)
    assert np.isclose(report["refinement_2_4_8"]["field_relative_L2"]["order"], 2)


def test_mcnemar_power_size_monotonicity_and_simulation():
    assert power.mcnemar_power(2000, 0.05, 0.0) <= 0.05 + 1e-12
    low, high = power.mcnemar_power(2000, 0.05, 0.005), power.mcnemar_power(2000, 0.05, 0.015)
    assert low < high
    rng = np.random.default_rng(0)
    n, pd, delta, rejections, trials = 400, 0.15, 0.06, 0, 400
    p10, p01 = (pd + delta)/2, (pd - delta)/2
    for _ in range(trials):
        cell = rng.choice(3, size=n, p=[p10, p01, 1 - pd])
        y = np.zeros(n, int)
        first = np.where(cell == 1, 1, 0)
        second = np.where(cell == 0, 1, 0)
        rejections += paired_comparison(y, first, second)["p_two_sided"] < 0.05
    assert abs(rejections/trials - power.mcnemar_power(n, pd, delta)) < 0.06
    assert power.minimum_detectable_effect(10000, 0.05) < power.minimum_detectable_effect(2500, 0.05)


def test_paired_tost():
    y = np.zeros(5000, int)
    same = power.paired_tost(y, y, y, margin=0.01)
    assert same["equivalent"]
    other = y.copy(); other[:500] = 1
    assert not power.paired_tost(y, y, other, margin=0.01)["equivalent"]


def test_photon_budget_limits():
    assert np.isclose(photon_budget.photon_energy(7.5e-4), 2.6486e-22, rtol=1e-3)
    assert 14.5 < photon_budget.thermal_occupation(7.5e-4, 300) < 15.7
    energy = np.array([[0.3, 0.1], [0.05, 0.2], [0.2, 0.1], [0.1, 0.15]])
    labels = np.array([0, 1, 0, 1])
    bright = photon_budget.counting_accuracy(energy, labels, 1e12, repeats=3)
    assert bright["accuracy"] == 1.0
    dark = photon_budget.counting_accuracy(energy, labels, 0.0, repeats=200, seed=4)
    assert 0.35 < dark["accuracy"] < 0.65
    quiet = photon_budget.nep_accuracy(energy, labels, 1.0, 0.0, 1e-3)
    assert quiet["accuracy"] == 1.0
    with pytest.raises(ValueError):
        photon_budget.counting_accuracy(energy*4, labels, 10)


def test_region_energy_export(tmp_path):
    n = 128
    optics = Optics(n=n, padding=2)
    model = DiffractiveNetwork(optics, np.ones((2, n, n))/2, seed=0)
    phases = tmp_path/"phases.npz"
    np.savez(phases, phases=model.phases, n=n, pixel_m=optics.pixel_m, wavelength_m=optics.wavelength_m,
             distances_m=optics.distances_m, padding=2, refinement=1, boundary="aperture")
    split = tmp_path/"dev.npz"
    np.savez(split, images=np.random.default_rng(0).uniform(0.1, 1, (3, 28, 28)), labels=[0, 1, 2], sample_ids=[5, 6, 7])
    energy = photon_budget.export_region_energy(phases, split, tmp_path/"energy.npz")
    assert energy.shape == (3, 10) and np.all(energy >= 0) and np.all(energy.sum(axis=1) <= 1 + 1e-9)
    with pytest.raises(ValueError):
        photon_budget.export_region_energy(phases, tmp_path/"test.npz", tmp_path/"x.npz")


def test_robustness_identity_and_bounds():
    n = 8
    model = DiffractiveNetwork(Optics(n=n, distances_m=(0.006,)*4, padding=2), small_masks(n), seed=2)
    x = smooth_inputs(n)
    reference = model.forward(x)["scores"]
    for kind, value in (("none", None), ("shift", [(0, 0)]*3), ("distance", [0, 0, 0, 0]),
                        ("wavelength", model.optics.wavelength_m)):
        scores = robustness.perturbed_scores(model, x, kind, value)
        if kind == "wavelength":  # phases are wrapped modulo 2*pi: exact up to rounding
            assert np.allclose(scores, reference, rtol=1e-10)
        else:
            assert np.allclose(scores, reference, rtol=1e-12, atol=1e-15)
    fine = robustness.perturbed_scores(model, x, "quantize", 4096)
    assert relative(fine, reference) < 1e-2
    with pytest.raises(ValueError):
        robustness.perturbed_scores(model, x, "shift", [(9, 0)]*3)
    with pytest.raises(ValueError):
        robustness.perturbed_scores(model, x, "phase_noise", 0.1)


def test_expressivity_linear_quadratic_and_rank():
    n = 6
    model = DiffractiveNetwork(Optics(n=n, distances_m=(0.006,)*3, padding=2), small_masks(n), beta=1.0, seed=4)
    a = expressivity.transfer_matrix(model)
    u = np.random.default_rng(0).normal(size=(n, n)) + 1j*np.random.default_rng(1).normal(size=(n, n))
    assert np.allclose(a @ u.ravel(), model.forward(u)["f"][model.roi].ravel(), atol=1e-12)
    q = expressivity.quadratic_forms(model)
    scores = np.real(np.einsum("i,kij,j->k", u.ravel().conj(), q, u.ravel()))
    assert np.allclose(scores, model.forward(u)["scores"], rtol=1e-10)
    assert all(np.linalg.eigvalsh(qk).min() > -1e-12 for qk in q)
    assert np.linalg.matrix_rank(q[0], tol=1e-10) <= 4
    rank = expressivity.jacobian_rank(model)
    assert rank["rank"] == rank["global_phase_bound"] == 2*n*n - 1


def test_score_gradients_match_loss_gradient_and_fd():
    n = 8
    model = DiffractiveNetwork(Optics(n=n, distances_m=(0.006,)*4, padding=2), small_masks(n), beta=10, seed=5)
    x, y = smooth_inputs(n, 2), np.array([0, 1])
    scores = model.forward(x)["scores"]
    p = np.exp(scores - scores.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    p[np.arange(2), y] -= 1
    assert np.allclose(expressivity.score_gradients(model, x, p/2), model.loss_and_grad(x, y)[1], rtol=1e-9, atol=1e-12)
    w = np.zeros((2, 2)); w[0, 1] = 1
    g = expressivity.score_gradients(model, x, w)
    index, h = (1, 3, 4), 1e-6
    base = model.phases[index]
    model.phases[index] = base + h; plus = model.forward(x)["scores"][0, 1]
    model.phases[index] = base - h; minus = model.forward(x)["scores"][0, 1]
    model.phases[index] = base
    assert np.isclose((plus - minus)/(2*h), g[index], rtol=1e-5, atol=1e-10)


def test_budget_estimate_and_measure():
    protocol = dict(epochs=2, seeds=[0, 1], beta_grid=[1], optical_lr_grid=[1])
    e = budget.estimate(protocol, seconds_per_sample=1e-3, train_size=1000, dev_size=100, test_size=100,
                        eval_cost_ratio=0.5)
    selection = 1*(2*1000*1e-3 + 2*100*1e-3*0.5)
    trained = 2*2*(2*1000*1e-3 + (2*100 + 100)*1e-3*0.5)
    untrained = 2*2*100*1e-3*0.5
    assert np.isclose(e["total_device_hours"], (selection + trained + untrained)/3600)
    m = budget.measure(Optics(n=128, padding=2), batch=2, steps=1)
    assert m["seconds_per_sample"] > 0 and m["m"] == 256


def _gz(path, payload):
    with gzip.open(path, "wb") as stream:
        stream.write(payload)


def test_qmnist_writer_split_and_formats(tmp_path):
    rng = np.random.default_rng(0)
    mnist, qdir = tmp_path/"mnist", tmp_path/"qmnist"
    mnist.mkdir(); qdir.mkdir()
    n_train, n_test, n_q = 60, 12, 30
    labels = rng.integers(0, 10, n_train).astype(np.uint8)
    test_labels = rng.integers(0, 10, n_test).astype(np.uint8)
    images = lambda k: struct.pack(">IIII", 2051, k, 28, 28) + rng.integers(0, 255, k*784, dtype=np.uint8).tobytes()
    _gz(mnist/"train-images-idx3-ubyte.gz", images(n_train))
    _gz(mnist/"train-labels-idx1-ubyte.gz", struct.pack(">II", 2049, n_train) + labels.tobytes())
    _gz(mnist/"t10k-images-idx3-ubyte.gz", images(n_test))
    _gz(mnist/"t10k-labels-idx1-ubyte.gz", struct.pack(">II", 2049, n_test) + test_labels.tobytes())
    writers = np.repeat(np.arange(12), 5)
    series = np.where(writers < 6, 0, 4)
    train_ext = np.zeros((n_train, 8), ">i4"); train_ext[:, 0] = labels; train_ext[:, 1] = series; train_ext[:, 2] = writers
    test_ext = np.zeros((n_test + n_q, 8), ">i4"); test_ext[:n_test, 0] = test_labels
    test_ext[n_test:, 0] = rng.integers(0, 10, n_q); test_ext[:, 1] = 4; test_ext[:, 2] = 100 + np.arange(n_test + n_q)//3
    _gz(qdir/"qmnist-train-labels-idx2-int.gz", struct.pack(">III", 3074, n_train, 8) + train_ext.tobytes())
    _gz(qdir/"qmnist-test-labels-idx2-int.gz", struct.pack(">III", 3074, n_test + n_q, 8) + test_ext.tobytes())
    _gz(qdir/"qmnist-test-images-idx3-ubyte.gz", images(n_test + n_q))
    mh = [digest_file(mnist/f) for f in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                                         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")]
    qh = {f: digest_file(qdir/f) for f in qmnist.QMNIST_HASHES}
    qmnist.prepare(mnist, qdir, tmp_path/"out", mnist_hashes=mh, qmnist_hashes=qh)
    train, dev = np.load(tmp_path/"out/train.npz"), np.load(tmp_path/"out/dev.npz")
    assert not set(train["writer_ids"]) & set(dev["writer_ids"])
    assert set(np.unique(dev["hsf_series"])) == {0, 4}
    assert len(np.load(tmp_path/"out/q50k.npz")["labels"]) == n_q
    with pytest.raises(ValueError):
        qmnist.prepare(mnist, qdir, tmp_path/"out2", qmnist_hashes={k: "0"*64 for k in qh}, mnist_hashes=mh)
    bad = tmp_path/"bad.gz"
    _gz(bad, struct.pack(">III", 3073, 1, 8) + b"\0"*32)
    with pytest.raises(ValueError):
        qmnist.read_idx2_int(bad)


def test_claim_register_and_language(tmp_path):
    evidence = tmp_path/"summary.json"; evidence.write_text("{}")
    register = dict(claims=[
        dict(id="H1", statement="D2NN mean accuracy >= 0.90", type="statistical", scope="scalar simulation",
             falsification="mean < 0.90", status="supported",
             evidence=[dict(artifact="summary.json", sha256=digest_file(evidence), kind="summary")]),
        dict(id="N1", statement="first protocol of its kind", type="novelty", scope="software",
             falsification="prior protocol found", status="supported",
             evidence=[dict(artifact="summary.json", sha256=digest_file(evidence), kind="summary")])])
    path = tmp_path/"claims.json"; path.write_text(json.dumps(register))
    problems = claims.check_register(path)
    assert any("prior-art" in p for p in problems) and not any(p.startswith("H1") for p in problems)
    previous = tmp_path/"old.json"
    previous.write_text(json.dumps(dict(claims=register["claims"] + [dict(id="R1")])))
    assert any("removed" in p for p in claims.check_register(path, previous))
    text = tmp_path/"paper.md"
    text.write_text("The D2NN outperforms the CNN.\nAccuracy target met [C:H1] and it outperforms chance.\n")
    findings = claims.audit_text(text, path)
    assert [f["line"] for f in findings] == [1]


def test_blind_roundtrip_and_tamper(tmp_path):
    results = tmp_path/"runs"
    for model in ("d2nn", "cnn"):
        for seed in (0, 1):
            (results/f"{model}_seed{seed}").mkdir(parents=True)
            np.savez(results/f"{model}_seed{seed}"/"predictions.npz", predictions=[seed])
    blinded, key = tmp_path/"blinded", tmp_path/"sealed"/"key.json"
    key.parent.mkdir()
    blind.blind(results, blinded, key, ["d2nn", "cnn"], [0, 1])
    assert not any("d2nn" in p.name for p in blinded.iterdir())
    analysis = tmp_path/"analysis.py"; analysis.write_text("print(1)\n")
    with pytest.raises(RuntimeError):
        blind.unblind(blinded, key)
    blind.freeze_analysis(blinded, [analysis])
    codes = blind.unblind(blinded, key)
    assert set(codes) == {"d2nn", "cnn"}
    analysis.write_text("print(2)\n")
    with pytest.raises(RuntimeError):
        blind.unblind(blinded, key)


def test_release_check_v2(tmp_path):
    script = ROOT/"scripts"/"release_check_v2.py"
    manifest = tmp_path/"closure.json"
    subprocess.run([sys.executable, str(script), str(manifest), "--init"], check=True, capture_output=True)
    failed = subprocess.run([sys.executable, str(script), str(manifest)], capture_output=True, text=True)
    assert failed.returncode == 1 and '"ready": false' in failed.stdout
    data = json.loads(manifest.read_text())
    artifact = tmp_path/"a.json"; artifact.write_text("{}")
    for gate in data["gates"].values():
        gate.update(status="passed", artifact="a.json", sha256=digest_file(artifact))
    manifest.write_text(json.dumps(data))
    passed = subprocess.run([sys.executable, str(script), str(manifest)], capture_output=True, text=True)
    assert passed.returncode == 0, passed.stdout


def test_protocol_manifest(tmp_path):
    sys.path.insert(0, str(ROOT/"scripts"))
    import protocol_manifest
    (tmp_path/"protocol.json").write_text("{}")
    manifest = protocol_manifest.build(["protocol.json"], tmp_path, "STUDY")
    assert manifest["files"]["protocol.json"] == digest_file(tmp_path/"protocol.json")
    with pytest.raises(FileNotFoundError):
        protocol_manifest.build(["../outside.json"], tmp_path, "STUDY")


def test_fresh_evaluation_end_to_end(tmp_path):
    sys.path.insert(0, str(ROOT/"scripts"))
    import smoke_run
    smoke_run.run(tmp_path/"smoke")
    rng = np.random.default_rng(3)
    fresh = tmp_path/"q50k.npz"
    np.savez(fresh, images=rng.uniform(0.02, 0.1, (20, 8, 8)), labels=np.arange(20) % 10,
             sample_ids=np.arange(1000, 1020), writer_ids=np.arange(20)//4, hsf_series=np.where(np.arange(20) < 10, 1, 4))
    frozen, runs = tmp_path/"smoke"/"selected.json", tmp_path/"smoke"/"runs"
    qmnist.fresh_evaluation(frozen, runs, fresh)
    with pytest.raises(FileExistsError):
        qmnist.fresh_evaluation(frozen, runs, fresh)
    result = qmnist.writer_grouped_summary(frozen, runs, fresh, "predictions_q50k.npz", tmp_path/"summary.json")
    assert set(result["comparisons"]) == {"linear", "mlp12", "mlp128", "cnn", "quadratic12"}
    assert result["comparisons"]["cnn"]["writer_grouped_bootstrap"]["grouping"] == "group"


def test_rs_frozen_comparison_small():
    n = 8
    optics = Optics(n=n, distances_m=(0.006,)*3, padding=16, refinement=3, boundary="aperture")
    phases = np.random.default_rng(6).uniform(0, 2*np.pi, (2, n, n))
    result = rs_network.frozen_comparison(phases, optics, small_masks(n), smooth_inputs(n, 2), quadratures=(1, 3))
    assert result["rs_quadrature"]["q1_vs_q3"] > 0 and result["core_vs_rs_finest"] < 0.2
    with pytest.raises(ValueError):
        rs_network.frozen_comparison(phases, Optics(n=n, distances_m=(0.006,)*3, padding=2, boundary="transparent"),
                                     small_masks(n), smooth_inputs(n, 1))


def test_guards_reserved_splits_and_frozen_optics(tmp_path):
    from neuro3d_completion.guards import frozen_optics, refuse_reserved_split
    images = np.zeros((1, 28, 28))
    for name in ("test.npz", "q50k.npz", "fresh_eval.npz"):
        np.savez(tmp_path/name, images=images)
        with pytest.raises(ValueError):
            refuse_reserved_split(tmp_path/name)
    data = tmp_path/"data"; data.mkdir()
    np.savez(data/"held.npz", images=images + 1)
    np.savez(data/"dev.npz", images=images)
    (data/"manifest.json").write_text(json.dumps({"values": {"split_sha256": {
        "test": digest_file(data/"held.npz"), "dev": digest_file(data/"dev.npz")}}}))
    with pytest.raises(ValueError):
        refuse_reserved_split(data/"held.npz")  # renamed copy of a reserved split
    assert refuse_reserved_split(data/"dev.npz") == data/"dev.npz"
    run = tmp_path/"run"; run.mkdir()
    np.savez(run/"phases.npz", phases=np.zeros((3, 16, 16)), pixel_m=4e-4, wavelength_m=7.5e-4,
             distances_m=[0.018]*4)
    optics, source = frozen_optics(run/"phases.npz", padding=4, refinement=3)
    assert (optics.n, optics.padding, optics.refinement, optics.filter_mode) == (16, 4, 3, "none")
    assert source.startswith("phases.npz")
    (run/"architecture.json").write_text(json.dumps({"optics": dict(
        n=16, pixel_m=4e-4, wavelength_m=7.5e-4, distances_m=[0.018]*4, padding=2, refinement=2,
        boundary="aperture", filter_mode="phase_gradient")}))
    optics, source = frozen_optics(run/"phases.npz", padding=4, refinement=3)
    assert (optics.padding, optics.refinement, optics.filter_mode, source) == (2, 2, "phase_gradient", "architecture.json")


def test_reproduction_script_fast_parts(tmp_path):
    script = ROOT/"scripts"/"reproduce_v2_review.py"
    done = subprocess.run([sys.executable, str(script), "--output", str(tmp_path), "--parts", "roughness", "ranks"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    rough = json.loads((tmp_path/"roughness.json").read_text())["values"]
    assert np.isclose(rough["uniform_expected"], np.pi/2) and len(rough["horizontal"]) == 3
    ranks = json.loads((tmp_path/"ranks.json").read_text())["values"]
    assert [ranks[k]["rank"] for k in ("L1_aperture", "L2_aperture", "L3_aperture", "L3_transparent")] == [36, 71, 106, 108]
