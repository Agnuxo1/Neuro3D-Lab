"""Generate an English manuscript draft and publication figures from real records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate(mnist_path, fashion_path, numerics_path, output):
    """Use admitted experiment values; reject synthetic or undersized benchmark records."""
    records = {"MNIST": json.loads(Path(mnist_path).read_text()),
               "Fashion-MNIST": json.loads(Path(fashion_path).read_text())}
    numerics = json.loads(Path(numerics_path).read_text())
    for record in records.values():
        if (record.get("sample_count") != 10000 or len(record.get("seeds", [])) < 10 or
                "SYNTHETIC" in record.get("study_id", "").upper()):
            raise ValueError("a scientific draft requires 10000 TEST samples and at least 10 real paired seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lines = ["# Finite-support validation and reproducible benchmarking of a scalar diffractive optical network",
             "", "Francisco Angulo de Lafuente", "", "## Abstract", "",
             "We examine a scalar phase-only diffractive network with fixed physical phase support. "
             "Computational padding and tied subpixel quadrature are independent numerical controls. "
             "Models are selected on DEV and evaluated on fixed official TEST sets using paired seeds. "
             "This manuscript reports simulations and does not establish a physical-device advantage.",
             "", "## 1. Background and contribution", "",
             "Diffractive neural networks have an experimental precedent in Lin et al. (2018). "
             "The contribution evaluated here is the finite-support numerical and reproducibility protocol. "
             "Originality beyond this protocol requires a separate literature assessment.",
             "", "## 2. Model and numerical conventions", "",
             r"The time convention is $e^{-i\omega t}$. Propagating scalar components use "
             r"$H(f_x,f_y)=\exp(2\pi i d\sqrt{\lambda^{-2}-f_x^2-f_y^2})$. "
             "Evanescent components are removed. Finite numerical windows remain periodic, "
             "so window and quadrature convergence must be evaluated independently.",
             "", "Computational refinement repeats each physical phase parameter across tied subpixels. "
             "The input is a cell-energy amplitude; detector scores sum cell energy in fixed physical regions.",
             "", "## 3. Dataset and evaluation protocol", "",
             "Official training files are split into the first 50000 TRAIN and remaining 10000 DEV images. "
             "The official 10000-image TEST file is reserved for frozen evaluations. Every model receives "
             "the same resized unit-energy input. Full predictions and sample IDs support paired statistics.",
             "", "## 4. Results", "",
             "| Dataset | Model | TEST mean | Sample SD across seeds | Minimum seed |",
             "|---|---|---:|---:|---:|"]
    for dataset, result in records.items():
        for name, values in result["models"].items():
            lines.append(f"| {dataset} | {name} | {values['mean']:.6f} | {values['sample_sd']:.6f} | {values['minimum']:.6f} |")
        names = list(result["models"])
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        for position, name in enumerate(names):
            values = result["models"][name]
            ax.scatter(np.full(len(values["per_seed"]), position), values["per_seed"], s=15, color="0.55")
            ax.errorbar(position, values["mean"], yerr=values["sample_sd"], fmt="o", color="#146A80", capsize=4)
        ax.set(xticks=range(len(names)), xticklabels=names, ylim=(0, 1), ylabel="TEST accuracy",
               title=f"{dataset}: mean and sample SD across paired seeds")
        ax.tick_params(axis="x", rotation=25)
        fig.savefig(output/(dataset+".png"), dpi=300)
        fig.savefig(output/(dataset+".pdf"))
        plt.close(fig)
    lines += ["", "## 5. Paired statistical comparisons", "",
              "McNemar tests use seed 0 and Holm correction over the five digital comparisons per dataset. "
              "Crossed bootstrap intervals resample paired seeds and common TEST examples. These empirical "
              "intervals do not replace an analysis of writer clustering when writer identifiers are unavailable."]
    for dataset, result in records.items():
        lines += ["", f"### {dataset}", "", f"The DEV-selected primary baseline is `{result['primary_baseline']}`.",
                  "", "```json", json.dumps(result["seed0_comparisons"], indent=2), "```",
                  "", "Registered verdicts: `"+json.dumps(result["verdicts"])+"`."]
    lines += ["", "## 6. Numerical validation", "", "The following record is reported without phase fitting or replacement of failed values:",
              "", "```json", json.dumps(numerics, indent=2), "```",
              "", "## 7. Error-budget interpretation", "",
              "Contractive exact-grid operators permit telescoping field-error bounds. A detector score changes "
              "by at most beta times (2 times reference field norm times field error plus squared field error). "
              "A label is certified only when the score margin exceeds twice a valid score-error bound. "
              "Differences between two grids are diagnostic estimates unless an additional remainder bound is proved.",
              "", "## 8. Limitations", "",
              "The model omits polarization, inter-plane reflections, dispersion, fabrication noise, "
              "and calibrated hardware. Equal epoch counts do not imply equal optimization effort or energy. "
              "All claims are restricted to the admitted datasets, seeds, numerical configuration, and supplied records.",
              "", "## 9. Reproducibility and data availability", "",
              "The artifact archive must include the frozen protocol, source and dataset hashes, environment locks, "
              "phase parameters, per-sample predictions, complete logs, and numerical validation. "
              "The hashes of records used to generate this draft are:", ""]
    for path in (mnist_path, fashion_path, numerics_path):
        path = Path(path)
        lines.append(f"- `{path.name}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`.")
    lines += ["", "## 10. Conclusions", "",
              "The registered results determine which accuracy and comparison hypotheses are supported. "
              "Failed superiority tests must remain negative results. No inference about physical speed, "
              "energy, or Nobel-level significance follows from this simulation benchmark.",
              "", "## References", "",
              "1. Lin et al. (2018). All-optical machine learning using diffractive deep neural networks. "
              "Science. https://doi.org/10.1126/science.aat8084",
              "2. Matsushima and Shimobaba (2009). Band-limited angular spectrum method for numerical "
              "simulation of free-space propagation in far and near fields. https://doi.org/10.1364/OE.17.019662",
              "", "Draft status: generated from supplied records; author and independent scientific review are required.", ""]
    (output/"manuscript.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist", type=Path, required=True)
    parser.add_argument("--fashion", type=Path, required=True)
    parser.add_argument("--numerics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.mnist, args.fashion, args.numerics, args.output)
    print(args.output/"manuscript.md")


if __name__ == "__main__":
    main()
