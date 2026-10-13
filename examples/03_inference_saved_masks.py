"""Example 3: classify DEV images with saved phase masks (TEST and Q50K splits are refused).

Run:
  python examples/03_inference_saved_masks.py --phases experiments/results/d2nn_seed0_phases.npz
         --split data/mnist-writer/dev.npz --limit 64
If the phases come from `benchmark final`, the optics are read from its architecture.json.
The historical seed-0 masks were trained on 8 x 8 digits: on MNIST they show the procedure only.
"""
import argparse
from pathlib import Path

import numpy as np

from neuro3d_completion.core import DiffractiveNetwork, detector_masks
from neuro3d_completion.data import encode
from neuro3d_completion.guards import frozen_optics, refuse_reserved_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=1)
    args = parser.parse_args()
    refuse_reserved_split(args.split)
    optics, source = frozen_optics(args.phases, padding=args.padding, refinement=args.refinement)
    with np.load(args.phases, allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    with np.load(args.split, allow_pickle=False) as data:
        images, labels = data["images"][:args.limit], data["labels"][:args.limit]
    x = encode(images)
    off = (optics.n - x.shape[-1])//2
    x = np.pad(x, ((0, 0), (off, off), (off, off)))
    model = DiffractiveNetwork(optics, detector_masks(optics.n, optics.pixel_m))
    model.phases = phases
    predictions = model.predict(x, batch_size=8)
    print(f"optics from {source}: padding {optics.padding}, refinement {optics.refinement}, boundary {optics.boundary}")
    print(f"accuracy on {len(labels)} DEV images: {np.mean(predictions == labels):.4f}")


if __name__ == "__main__":
    main()
