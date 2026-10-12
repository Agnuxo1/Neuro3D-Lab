"""Compare the legacy periodic optlab model with the reference implementation."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from optlab.d2nn import DiffractiveNetwork as LegacyNetwork
from neuro3d_completion.core import Optics, DiffractiveNetwork, detector_masks
from neuro3d_completion.evidence import record


def check():
    """Check complete complex fields, scores, loss, and gradient on four fixed inputs."""
    o = Optics(padding=1, refinement=1, boundary="periodic")
    masks = detector_masks(o.n, o.pixel_m)
    legacy = LegacyNetwork(o.n, o.pixel_m, o.wavelength_m, o.distances_m, masks, scale=100, seed=0)
    current = DiffractiveNetwork(o, masks, beta=100, seed=0)
    rng = np.random.default_rng(2026)
    inputs = rng.normal(size=(4, o.n, o.n))+1j*rng.normal(size=(4, o.n, o.n))
    inputs /= np.linalg.norm(inputs, axis=(1, 2))[:, None, None]
    labels = np.array([0, 3, 6, 9])
    outputs = [legacy.forward(u) for u in inputs]
    new = current.forward(inputs)
    legacy_losses, legacy_gradients = zip(*[legacy.loss_and_grad(u, int(y)) for u, y in zip(inputs, labels)])
    loss, grad = current.loss_and_grad(inputs, labels)
    def relative(a, b):
        return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))
    values = dict(field_relative=relative(new["f"], np.stack([v["f"] for v in outputs])),
                  intensity_relative=relative(new["I"], np.stack([v["I"] for v in outputs])),
                  scores_relative=relative(new["scores"], np.stack([v["scores"] for v in outputs])),
                  loss_absolute=abs(loss-float(np.mean(legacy_losses))),
                  gradient_relative=relative(grad, np.mean(legacy_gradients, axis=0)))
    values["pass"] = (max(values["field_relative"], values["intensity_relative"], values["scores_relative"]) <= 1e-9
                      and values["loss_absolute"] <= 1e-10 and values["gradient_relative"] <= 1e-8)
    return o, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    optics, result = check()
    record(args.output, "legacy periodic implementation compatibility", optics, result)
    print(result)
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
