"""Immutable records, exact paired statistics, and conditional error budgets."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
from scipy.stats import binomtest


def digest_file(path):
    """Hash exact bytes without interpreting dataset labels or predictions."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def code_digest():
    """Fingerprint every Python file in this reference package."""
    root = Path(__file__).parent
    return {p.name: digest_file(p) for p in sorted(root.glob("*.py"))}


def environment():
    """Record the actual interpreter and installed scientific dependencies."""
    versions = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                code_sha256=code_digest())


def write_json_new(path, payload):
    """Create an immutable JSON artifact, refusing replacement of earlier evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def record(path, kind, configuration, values):
    """Write a complete dated record with configuration and code provenance."""
    configuration = asdict(configuration) if hasattr(configuration, "__dataclass_fields__") else configuration
    write_json_new(path, dict(kind=kind, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                             configuration=configuration, values=values, environment=environment()))


def reserve_test(ledger, study_id, model_id, seed, test_path, frozen_config):
    """Reserve TEST before loading it; a failed evaluation remains consumed.

    The key intentionally excludes configuration: changing hyperparameters cannot
    obtain another TEST evaluation for the same study, model, seed, and dataset.
    This prevents accidental reuse, not deliberate protocol circumvention.
    """
    key_payload = [study_id, model_id, int(seed), digest_file(test_path)]
    key = hashlib.sha256(json.dumps(key_payload).encode()).hexdigest()
    target = Path(ledger) / (key + ".json")
    write_json_new(target, dict(identity=key_payload, config_sha256=digest_file(frozen_config),
                               timestamp_utc=datetime.now(timezone.utc).isoformat()))
    return target


def wilson(correct, count, z=1.959963984540054):
    """Return a Wilson interval for one model on one set of independent samples."""
    if not isinstance(count, (int, np.integer)) or count < 1 or correct < 0 or correct > count:
        raise ValueError("valid binomial counts are required")
    p, a = correct/count, z*z/count
    center = (p + a/2)/(1+a)
    radius = z*np.sqrt(p*(1-p)/count + z*z/(4*count*count))/(1+a)
    return [float(center-radius), float(center+radius)]


def paired_comparison(y, first, second):
    """Report accuracy difference and exact two-sided McNemar evidence."""
    y, first, second = map(np.asarray, (y, first, second))
    if y.ndim != 1 or len(y) == 0 or first.shape != y.shape or second.shape != y.shape:
        raise ValueError("paired nonempty prediction vectors are required")
    a, b = first == y, second == y
    first_only, second_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    discordant = first_only + second_only
    p = 1.0 if discordant == 0 else float(binomtest(first_only, discordant, 0.5).pvalue)
    return dict(first_accuracy=float(a.mean()), second_accuracy=float(b.mean()),
                difference=float(a.mean()-b.mean()), first_only=first_only,
                second_only=second_only, p_two_sided=p)


def holm(p_values):
    """Adjust one predeclared family of p-values by Holm's step-down method."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.any(p < 0) or np.any(p > 1) or not np.isfinite(p).all():
        raise ValueError("finite p-values in [0,1] are required")
    order = np.argsort(p)
    adjusted = np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p))))
    result = np.empty_like(p)
    result[order] = np.minimum(1, adjusted)
    return result.tolist()


def crossed_bootstrap(y, first, second, repetitions=2000, seed=2026, group_ids=None):
    """Resample paired seeds and examples/groups, preserving shared TEST dependence.

    Predictions have shape (seeds,samples) and seeds are paired by the protocol.
    If writer IDs exist, resample writers instead of individual examples. This
    is an empirical uncertainty estimate and not a universal coverage guarantee.
    """
    y, first, second = map(np.asarray, (y, first, second))
    if (y.ndim != 1 or len(y) == 0 or first.ndim != 2 or first.shape != second.shape
            or first.shape[1] != len(y) or first.shape[0] < 2):
        raise ValueError("at least two paired seeds and common nonempty samples are required")
    if not isinstance(repetitions, int) or repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    diff = (first == y).astype(float) - (second == y).astype(float)
    if group_ids is None:
        groups = [np.array([i]) for i in range(len(y))]
    else:
        group_ids = np.asarray(group_ids)
        if group_ids.shape != y.shape:
            raise ValueError("group IDs must align with samples")
        groups = [np.flatnonzero(group_ids == g) for g in np.unique(group_ids)]
    rng, draws = np.random.default_rng(seed), []
    for _ in range(repetitions):
        seeds = rng.integers(diff.shape[0], size=diff.shape[0])
        if group_ids is None:
            samples = rng.integers(len(y), size=len(y))
        else:
            samples = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        draws.append(float(diff[np.ix_(seeds, samples)].mean()))
    return dict(mean_difference=float(diff.mean()), percentile_95=np.quantile(draws, [.025, .975]).tolist(),
                repetitions=repetitions, grouping="sample" if group_ids is None else "group")


def contraction_budget(input_norm, input_error, operator_errors, phase_errors,
                       residual_errors=None):
    """Propagate supplied valid operator/phase error bounds through contractions.

    Bounds must refer to the same space, coordinates, and norm. Numerical
    differences observed between two grids are not automatically valid bounds.
    """
    norm, error = float(input_norm), float(input_error)
    op, phase = np.asarray(operator_errors, float), np.asarray(phase_errors, float)
    residual = np.zeros_like(op) if residual_errors is None else np.asarray(residual_errors, float)
    if (op.ndim != 1 or len(op) != len(phase)+1 or residual.shape != op.shape or
            norm < 0 or error < 0 or not np.isfinite([norm, error]).all() or
            not np.isfinite(np.r_[op, phase, residual]).all() or
            np.any(np.r_[op, phase, residual] < 0)):
        raise ValueError("finite nonnegative consistent error budgets are required")
    for k, eta in enumerate(op):
        error += eta*norm + residual[k]
        if k < len(phase):
            error += min(2.0, phase[k])*norm
    return float(error)
