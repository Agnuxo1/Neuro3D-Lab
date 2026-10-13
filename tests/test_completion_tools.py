"""The sharded DEV selection must reproduce benchmark.select exactly on synthetic data."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path):
    from neuro3d_completion.core import Optics
    from neuro3d_completion.data import save_split
    from neuro3d_completion.evidence import write_json_new
    rng = np.random.default_rng(2026)
    for split, count, offset in (("train", 20, 0), ("dev", 10, 20), ("test", 10, 0)):
        images = rng.uniform(.02, .1, (count, 8, 8))
        labels = np.arange(count) % 10
        for i, label in enumerate(labels):
            images[i, label % 8, :] += .5
        save_split(tmp_path/"data"/(split + ".npz"), images, labels, np.arange(offset, offset + count))
    protocol = dict(study_id="SYNTHETIC-SHARD-CHECK", data_dir="data", epochs=1, batch=4, seeds=[0, 1],
                    optics=asdict(Optics(padding=2)), beta_grid=[100, 200], optical_lr_grid=[.02],
                    digital_lr_grid=[.001, .003],
                    hypotheses=dict(mean_accuracy_min=.9, seed_accuracy_min=.88, alpha=.05,
                                    permuted_accuracy_max=.15, untrained_accuracy_max=.2, no_layer_accuracy_max=.2))
    path = tmp_path/"protocol.json"
    write_json_new(path, protocol)
    return path


def test_sharded_selection_matches_select(tmp_path):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT/"scripts"/"completion_tools"))
    import select_sharded
    from neuro3d_completion.benchmark import MODELS, select
    torch.set_num_threads(1)
    protocol = _fixture(tmp_path)
    select(protocol, tmp_path/"reference.json", "cpu")
    shards = tmp_path/"shards"
    for name in MODELS:
        for index in range(len(select_sharded.grid(json.loads(protocol.read_text()), name))):
            select_sharded.shard(protocol, shards, name, index, "cpu")
    with pytest.raises(FileExistsError):
        select_sharded.shard(protocol, shards, "linear", 0, "cpu")
    select_sharded.assemble(protocol, shards, tmp_path/"sharded.json", "cpu")
    reference = json.loads((tmp_path/"reference.json").read_text())
    sharded = json.loads((tmp_path/"sharded.json").read_text())
    for key in ("selected", "selections", "best_baseline_on_dev", "data_sha256", "protocol_file_sha256"):
        assert sharded[key] == reference[key], key
    assert sharded["environment"]["code_sha256"] == reference["environment"]["code_sha256"]
    torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize("refinement", [1, 2])
def test_lean_fidelity_model_matches_torch_backend(refinement):
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT/"scripts"/"completion_tools"))
    import fidelity_study
    from neuro3d_completion.core import DiffractiveNetwork, Optics, detector_masks
    from neuro3d_completion.torch_backend import TorchD2NN
    o = Optics(n=128, padding=2, refinement=refinement)
    reference = DiffractiveNetwork(o, detector_masks(o.n, o.pixel_m), beta=100.0, seed=3)
    standard = TorchD2NN(reference)
    lean = fidelity_study.lean_model(reference.phases, o, 100.0, "cpu")
    x = torch.rand(2, 128, 128, dtype=torch.float64)
    a, b = standard(x), lean(x)
    assert torch.allclose(a, b, rtol=1e-12, atol=0)


def test_q50k_verdicts_and_window_check(tmp_path):
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT/"scripts"))
    sys.path.insert(0, str(ROOT/"scripts"/"completion_tools"))
    import smoke_run
    import q50k_verdicts
    import window_check
    from neuro3d_completion import qmnist
    smoke_run.run(tmp_path/"smoke")
    rng = np.random.default_rng(3)
    fresh = tmp_path/"q50k.npz"
    np.savez(fresh, images=rng.uniform(0.02, 0.1, (20, 8, 8)), labels=np.arange(20) % 10,
             sample_ids=np.arange(1000, 1020), writer_ids=np.arange(20)//4, hsf_series=np.where(np.arange(20) < 10, 1, 4))
    frozen, runs = tmp_path/"smoke"/"selected.json", tmp_path/"smoke"/"runs"
    qmnist.fresh_evaluation(frozen, runs, fresh)
    result = q50k_verdicts.verdicts(frozen, runs, fresh, "predictions_q50k.npz")
    assert set(result["seed0_comparisons"]) == {"linear", "mlp12", "mlp128", "cnn", "quadratic12"}
    assert all("holm_p" in c and c["writer_grouped_bootstrap"]["grouping"] == "group"
               for c in result["seed0_comparisons"].values())
    assert isinstance(result["H2_writer_grouped"], bool) and result["writers"] == 5
    run = tmp_path/"run"
    run.mkdir()
    scores = rng.uniform(size=(30, 10))
    labels, ids = np.arange(30) % 10, np.arange(30)
    np.savez(run/"eval_r1_p4_scores.npz", scores=scores, labels=labels, sample_ids=ids)
    np.savez(run/"eval_r1_p8_scores.npz", scores=scores*(1 + 1e-4), labels=labels, sample_ids=ids)
    w1 = window_check.compare(run, "r1_p4", "r1_p8")
    assert w1["changed_label_fraction"] == 0 and abs(w1["score_relative_L2"] - 1e-4) < 1e-9
