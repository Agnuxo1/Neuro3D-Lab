"""Write the ordered GPU and CPU job lists of CONTRATO-013 and CONTRATO-014 for queue_driver.py."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Interpreter of the job commands: $NEURO3D_PYTHON, else the interpreter running this script.
PY = os.environ.get("NEURO3D_PYTHON", sys.executable)
DIGITAL = ("linear", "mlp12", "mlp128", "cnn", "quadratic12")
OPTICAL = ("d2nn", "permuted", "untrained", "no_layers")
SEEDS = range(10)
DATASETS = (("mnist-writer", "completion-configs/closure_mnist_writer_p4r1.json"),
            ("fashion", "completion-configs/closure_fashion_p4r1.json"))
FIDELITY_TRAIN = ((1, 2), (1, 4), (2, 2), (4, 2), (8, 2))
FIDELITY_EVAL = ((1, 2), (1, 4), (2, 2), (4, 2), (8, 2), (16, 2))
FID = "completion_results/fidelity"


def py(*args):
    return [PY, "-B", *args]


def gpu(name, cmd, done, vram=4, ram=3, requires=()):
    return dict(name=name, cmd=cmd, kind="gpu", done=done, vram_gb=vram, ram_gb=ram, requires=list(requires))


def cpu(name, cmd, done, ram=3, requires=()):
    return dict(name=name, cmd=cmd, kind="cpu", done=done, ram_gb=ram, requires=list(requires))


def fidelity_train(r, p, seed):
    run = f"{FID}/train_r{r}_p{p}_seed{seed}"
    vram = {1: 3, 2: 3, 4: 4, 8: 10}[r]
    return gpu(f"fid-train-r{r}p{p}s{seed}", py("scripts/completion_tools/fidelity_study.py", "train", "--refinement",
                                               str(r), "--padding", str(p), "--seed", str(seed)),
               f"{run}/train.json", vram=vram)


def fidelity_eval(tr, tp, seed, er, ep):
    run = f"{FID}/train_r{tr}_p{tp}_seed{seed}"
    vram = {1: 2, 2: 3, 4: 4, 8: 6, 16: 12}[er]
    return gpu(f"fid-eval-r{tr}p{tp}s{seed}-r{er}p{ep}",
               py("scripts/completion_tools/fidelity_study.py", "evaluate", "--run", run,
                  "--refinement", str(er), "--padding", str(ep)),
               f"{run}/eval_r{er}_p{ep}.json", vram=vram, requires=[f"{run}/train.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("completion_results/jobs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    gpu_jobs, cpu_jobs = [], []
    # Fast fidelity training first: early, cheap, DEV-only evidence.
    for r, p in FIDELITY_TRAIN[:3]:
        gpu_jobs += [fidelity_train(r, p, s) for s in range(3)]
    # Sharded DEV selection, then freezing, per dataset.
    for name, protocol in DATASETS:
        shards = f"completion_results/{name}-shards"
        grid = json.loads(Path(protocol).read_text(encoding="utf-8"))
        sizes = dict(d2nn=len(grid["beta_grid"])*len(grid["optical_lr_grid"]), **{m: len(grid["digital_lr_grid"]) for m in DIGITAL})
        for model in ("d2nn",) + DIGITAL:
            for index in range(sizes[model]):
                gpu_jobs.append(gpu(f"{name}-select-{model}-{index}",
                                    py("scripts/completion_tools/select_sharded.py", "shard", "--protocol", protocol,
                                       "--shards", shards, "--model", model, "--index", str(index), "--device", "cuda"),
                                    f"{shards}/{model}_{index}.json"))
        frozen = f"evidence/{name}-selected.json"
        gpu_jobs.append(gpu(f"{name}-assemble", py("scripts/completion_tools/select_sharded.py", "assemble", "--protocol",
                                                   protocol, "--shards", shards, "--frozen", frozen, "--device", "cuda"),
                            frozen))
    # Final runs: optical on the GPU, digital on the CPU.
    for name, _ in DATASETS:
        frozen, out = f"evidence/{name}-selected.json", f"completion_results/{name}"
        for model in OPTICAL:
            for seed in SEEDS:
                gpu_jobs.append(gpu(f"{name}-final-{model}-{seed}",
                                    py("-m", "neuro3d_completion.benchmark", "final", "--frozen", frozen, "--output-dir",
                                       out, "--model", model, "--seed", str(seed), "--device", "cuda"),
                                    f"{out}/{model}_seed{seed}/result.json", requires=[frozen]))
        for model in DIGITAL:
            for seed in SEEDS:
                cpu_jobs.append(cpu(f"{name}-final-{model}-{seed}",
                                    py("-m", "neuro3d_completion.benchmark", "final", "--frozen", frozen, "--output-dir",
                                       out, "--model", model, "--seed", str(seed), "--device", "cpu"),
                                    f"{out}/{model}_seed{seed}/result.json", requires=[frozen]))
    # Remaining fidelity work.
    for r, p in FIDELITY_TRAIN[3:]:
        gpu_jobs += [fidelity_train(r, p, s) for s in range(3)]
    for tr, tp in FIDELITY_TRAIN:
        for s in range(3):
            gpu_jobs += [fidelity_eval(tr, tp, s, er, ep) for er, ep in FIDELITY_EVAL]
    lean_run = f"{FID}/train_r8_p2_seed0"
    gpu_jobs.insert(len(gpu_jobs) - 3*len(FIDELITY_TRAIN)*len(FIDELITY_EVAL),
                    gpu("fid-lean-check", py("scripts/completion_tools/fidelity_study.py", "lean-check", "--run", lean_run),
                        f"{lean_run}/lean_check_r8_p2.json", vram=8, requires=[f"{lean_run}/train.json"]))
    for tr, tp in FIDELITY_TRAIN:
        run = f"{FID}/train_r{tr}_p{tp}_seed0"
        gpu_jobs.append(gpu(f"fid-rs-r{tr}p{tp}", py("scripts/completion_tools/fidelity_study.py", "rs", "--run", run),
                            f"{run}/rs_direct.json", vram=6, requires=[f"{run}/train.json"]))
    # CONTRATO-013 W1 and refinement line on the seed-0 D2NN masks (DEV only).
    for name, _ in DATASETS:
        run = f"completion_results/{name}/d2nn_seed0"
        for er, ep in ((1, 4), (1, 8), (2, 2), (4, 2), (8, 2)):
            gpu_jobs.append(gpu(f"{name}-w1-r{er}p{ep}", py("scripts/completion_tools/fidelity_study.py", "evaluate",
                                                            "--data", f"data/{name}", "--run", run,
                                                            "--refinement", str(er), "--padding", str(ep)),
                                f"{run}/eval_r{er}_p{ep}.json", vram=6, requires=[f"{run}/result.json"]))
        gpu_jobs.append(cpu(f"{name}-w1-check", py("scripts/completion_tools/window_check.py", "--run", run,
                                                   "--output", f"evidence/{name}-window-check.json"),
                            f"evidence/{name}-window-check.json", ram=1, requires=[f"{run}/eval_r8_p2.json"]))
    # Summaries once every final run of a dataset exists (verified frozen code and data first).
    for name, _ in DATASETS:
        frozen, out = f"evidence/{name}-selected.json", f"completion_results/{name}"
        runs = [f"{out}/{m}_seed{s}/result.json" for m in OPTICAL + DIGITAL for s in SEEDS]
        gpu_jobs.append(cpu(f"{name}-verify-summary", py("scripts/completion_tools/verify_frozen.py", "--frozen", frozen,
                                                         "--record", f"evidence/{name}-verify-before-summary.json"),
                            f"evidence/{name}-verify-before-summary.json", ram=2, requires=runs))
        gpu_jobs.append(cpu(f"{name}-summarize", py("-m", "neuro3d_completion.benchmark", "summarize", "--frozen", frozen,
                                                    "--output-dir", out, "--output", f"evidence/{name}-summary.json"),
                            f"evidence/{name}-summary.json", ram=2, requires=runs))
    gpu_jobs.append(cpu("mnist-writer-test10k-writer-verdicts",
                        py("scripts/completion_tools/q50k_verdicts.py", "--frozen", "evidence/mnist-writer-selected.json",
                           "--results", "completion_results/mnist-writer", "--split", "data/mnist-writer/test.npz",
                           "--output", "evidence/mnist-writer-test10k-writer-grouped.json"),
                        "evidence/mnist-writer-test10k-writer-grouped.json", ram=2))
    # Fresh test once every MNIST final run exists.
    mnist_runs = [f"completion_results/mnist-writer/{m}_seed{s}/result.json" for m in OPTICAL + DIGITAL for s in SEEDS]
    gpu_jobs.append(cpu("mnist-writer-verify-q50k", py("scripts/completion_tools/verify_frozen.py", "--frozen",
                                                       "evidence/mnist-writer-selected.json", "--q50k",
                                                       "data/mnist-writer/q50k.npz", "--record",
                                                       "evidence/mnist-writer-verify-before-q50k.json"),
                        "evidence/mnist-writer-verify-before-q50k.json", ram=2, requires=mnist_runs))
    gpu_jobs.append(gpu("mnist-writer-q50k", py("-m", "neuro3d_completion.qmnist", "evaluate", "--frozen",
                                                "evidence/mnist-writer-selected.json", "--results",
                                                "completion_results/mnist-writer", "--q50k", "data/mnist-writer/q50k.npz",
                                                "--device", "cuda"),
                        "completion_results/mnist-writer/no_layers_seed9/result_q50k.json", vram=4, ram=6,
                        requires=mnist_runs + ["evidence/mnist-writer-verify-before-q50k.json"]))
    gpu_jobs.append(cpu("mnist-writer-q50k-verdicts",
                        py("scripts/completion_tools/q50k_verdicts.py", "--frozen", "evidence/mnist-writer-selected.json",
                           "--results", "completion_results/mnist-writer", "--split", "data/mnist-writer/q50k.npz",
                           "--predictions", "predictions_q50k.npz", "--output", "evidence/mnist-writer-q50k-verdicts.json"),
                        "evidence/mnist-writer-q50k-verdicts.json", ram=3))
    gpu_jobs.append(cpu("mnist-writer-q50k-summary",
                        py("-m", "neuro3d_completion.qmnist", "summarize", "--frozen", "evidence/mnist-writer-selected.json",
                           "--results", "completion_results/mnist-writer", "--split", "data/mnist-writer/q50k.npz",
                           "--predictions", "predictions_q50k.npz", "--output", "evidence/q50k-writer-summary.json"),
                        "evidence/q50k-writer-summary.json", ram=3))
    gpu_jobs.append(cpu("fidelity-summary", py("scripts/completion_tools/fidelity_study.py", "summarize",
                                               "--output", "evidence/fidelity-summary.json"),
                        "evidence/fidelity-summary.json", ram=1))
    (args.out/"gpu_jobs.json").write_text(json.dumps(gpu_jobs, indent=1), encoding="utf-8")
    (args.out/"cpu_jobs.json").write_text(json.dumps(cpu_jobs, indent=1), encoding="utf-8")
    print(f"gpu jobs {len(gpu_jobs)}, cpu jobs {len(cpu_jobs)}")


if __name__ == "__main__":
    main()
