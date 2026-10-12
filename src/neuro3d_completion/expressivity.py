"""What a passive phase-only network can represent, measured rather than assumed.

Facts checked here numerically:

1. For fixed phases the detector-plane field is linear in the input: f = A u.
2. Every score is a positive semidefinite quadratic form s_k(u) = beta u^H Q_k u with
   Q_k = A^H M_k A, rank(Q_k) <= number of numerical samples in detector k.
3. With the absorbing ('aperture') exterior, a common phase offset on one layer only
   multiplies A by a unit scalar, so the real Jacobian of A with respect to the L*n^2
   phases has rank <= L*n^2 - L + 1, and also <= 2 * dim(input) * dim(output).
4. Pixels whose phase barely affects any score are not effective parameters; their
   fraction is measured by the per-pixel score sensitivity.

These are properties of the scalar model. They bound representational capacity; they are
not accuracy results. Related analyses: Kulce et al., Light Sci. Appl. 10, 25 (2021),
DOI 10.1038/s41377-020-00439-9, and 10, 196 (2021), DOI 10.1038/s41377-021-00623-5.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .core import DiffractiveNetwork, Optics, detector_masks
from .evidence import digest_file, record
from .guards import frozen_optics, refuse_reserved_split


def transfer_matrix(model):
    """Columns are detector-plane fine fields (roi, flattened) for each physical input pixel."""
    n = model.optics.n
    basis = np.eye(n*n).reshape(n*n, n, n)
    columns = []
    for start in range(0, n*n, 64):
        out = model.forward(basis[start:start+64])["f"]
        columns.append(out[(slice(None), *model.roi)].reshape(len(out), -1))
    return np.concatenate(columns).T


def quadratic_forms(model):
    """Q_k = A^H M_k A on the physical input space, with M_k the fine detector weights."""
    a = transfer_matrix(model)
    weights = model.full_masks[(slice(None), *model.roi)].reshape(len(model.masks), -1)
    return np.asarray([a.conj().T @ (w[:, None]*a) for w in weights])


def score_gradients(model, inputs, weights):
    """Gradient of sum_b sum_k weights[b,k] * s_k(u_b) with respect to all phases (adjoint)."""
    c = model.forward(inputs, cache=True)
    w = np.asarray(weights, float)
    if w.shape != c["scores"].shape:
        raise ValueError("weights must have shape (batch, classes)")
    g = model.beta*np.einsum("bk,kij->bij", w, model.full_masks)*c["f"]
    if model.optics.boundary == "aperture":
        g *= model.aperture
    g = model.propagate(g, model.kernels[-1], adjoint=True)
    gradient = np.empty_like(model.phases)
    o = model.optics
    for k in reversed(range(o.layers)):
        full = -2*np.imag(np.conj(g)*c["states"][k])
        gradient[k] = full[(slice(None), *model.roi)].reshape(len(w), o.n, o.refinement, o.n,
                                                              o.refinement).sum(axis=(0, 2, 4))
        g = model.propagate(g*np.conj(c["t"][k]), model.kernels[k], adjoint=True)
    return gradient


def pixel_influence(model, inputs):
    """Root-sum-square of every score's phase gradient over inputs, per physical pixel."""
    x = np.asarray(inputs)
    if x.ndim != 3 or len(x) == 0:
        raise ValueError("a nonempty (batch, n, n) input array is required")
    total = np.zeros_like(model.phases)
    classes = len(model.masks)
    for b in range(len(x)):
        for k in range(classes):
            w = np.zeros((1, classes)); w[0, k] = 1
            total += score_gradients(model, x[b:b+1], w)**2
    influence = np.sqrt(total)
    peak = influence.max()
    thresholds = {f"below_{t:g}_of_max": float(np.mean(influence < t*peak)) for t in (1e-2, 1e-3, 1e-6)}
    return influence, thresholds


def jacobian_rank(model, step=1e-6, tolerance=1e-7):
    """Numerical rank of d vec(A) / d phases (real Jacobian) by central differences."""
    base = model.phases.copy()
    columns = []
    for index in np.ndindex(base.shape):
        model.phases = base.copy(); model.phases[index] += step
        plus = transfer_matrix(model)
        model.phases = base.copy(); model.phases[index] -= step
        minus = transfer_matrix(model)
        d = (plus - minus).ravel()/(2*step)
        columns.append(np.concatenate([d.real, d.imag]))
    model.phases = base
    jac = np.asarray(columns).T
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > tolerance*singular[0]))
    o = model.optics
    return dict(rank=rank, phases=int(base.size), layers=o.layers,
                global_phase_bound=int(base.size - o.layers + 1),
                linear_map_real_dimension=int(2*(o.n*o.n)*(o.n*o.refinement)**2),
                singular_values_tail=singular[-min(5, len(singular)):].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="TRAIN or DEV split; reserved splits are refused")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--padding", type=int, default=2, help="used only without architecture.json")
    parser.add_argument("--refinement", type=int, default=1, help="used only without architecture.json")
    parser.add_argument("--boundary", choices=["aperture", "transparent"], default="aperture")
    parser.add_argument("--filter-mode", choices=["none", "phase_gradient"], default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refuse_reserved_split(args.split)
    from .data import encode
    with np.load(args.phases, allow_pickle=False) as data:
        phases = np.asarray(data["phases"], float)
    with np.load(args.split, allow_pickle=False) as data:
        images = data["images"][:args.limit]
    optics, source = frozen_optics(args.phases, args.padding, args.refinement, args.boundary, args.filter_mode)
    n = optics.n
    x = encode(images)
    off = (n - x.shape[-1])//2
    x = np.pad(x, ((0, 0), (off, off), (off, off)))
    model = DiffractiveNetwork(optics, detector_masks(n, optics.pixel_m))
    model.phases = phases
    influence, fractions = pixel_influence(model, x)
    result = dict(fractions=fractions, per_layer_median=np.median(influence.reshape(len(phases), -1), axis=1).tolist(),
                  phases_sha256=digest_file(args.phases), split_sha256=digest_file(args.split), optics_source=source,
                  scope="sensitivity of scores to each physical phase pixel on supplied TRAIN/DEV inputs")
    record(args.output, "phase-pixel influence", optics, result)
    print(args.output)


if __name__ == "__main__":
    main()
