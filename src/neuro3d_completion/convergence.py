"""Observed order and Grid Convergence Index (GCI) for refinement studies.

Implements the three-level procedure of Celik, Ghia, Roache and Freitas (2008),
J. Fluids Eng. 130(7), 078001, DOI 10.1115/1.2960953, for a constant refinement ratio:

    p   = |ln|eps32/eps21|| / ln r,   eps32 = phi3 - phi2, eps21 = phi2 - phi1
    GCI = 1.25 |(phi1 - phi2)/phi1| / (r^p - 1)

phi1 is the finest level. The estimate is meaningful only in the asymptotic range
(monotone convergence, observed order near the formal order). Oscillatory or
divergent sequences are reported as such and yield no GCI.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import numpy as np


def observed_order_from_differences(e_coarse, e_fine, ratio):
    """Observed order from two successive difference norms ||q1-q2||, ||q2-q3||."""
    e1, e2, r = float(e_coarse), float(e_fine), float(ratio)
    if not (math.isfinite(e1) and math.isfinite(e2) and r > 1):
        raise ValueError("finite differences and refinement ratio > 1 are required")
    if e1 <= 0 or e2 <= 0:
        return dict(status="degenerate", order=None, remaining_estimate=None)
    if e2 >= e1:
        return dict(status="not converging", order=None, remaining_estimate=None)
    p = math.log(e1/e2)/math.log(r)
    return dict(status="monotone", order=p, remaining_estimate=e2/(r**p - 1), gci=1.25*e2/(r**p - 1))


def gci_three_level(phi_fine, phi_medium, phi_coarse, ratio, safety=1.25):
    """Vectorised Celik et al. procedure for scalars or arrays (e.g. every detector score)."""
    f1, f2, f3 = (np.asarray(x, dtype=float) for x in (phi_fine, phi_medium, phi_coarse))
    r = float(ratio)
    if f1.shape != f2.shape or f2.shape != f3.shape or r <= 1:
        raise ValueError("equal shapes and refinement ratio > 1 are required")
    if not np.isfinite(np.r_[f1.ravel(), f2.ravel(), f3.ravel()]).all():
        raise ValueError("finite values are required")
    e21, e32 = f2 - f1, f3 - f2
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = e32/e21
        order = np.abs(np.log(np.abs(quotient)))/math.log(r)
        relative_error = np.abs((f1 - f2)/f1)
        gci = safety*relative_error/(r**order - 1)
        extrapolated = (r**order*f1 - f2)/(r**order - 1)
    monotone = (quotient > 0) & (np.abs(e32) > np.abs(e21)) & (e21 != 0)
    status = np.where(e21 == 0, "converged-or-degenerate",
                      np.where(quotient < 0, "oscillatory",
                               np.where(monotone, "monotone", "divergent")))
    invalid = status != "monotone"
    order, gci, extrapolated = (np.where(invalid, np.nan, x) for x in (order, gci, extrapolated))
    return dict(order=order, gci_fine=gci, extrapolated=extrapolated, status=status)


def from_record(path):
    """Read a validation record with keys like 'refinement_2_vs_4' and report observed orders."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("values", payload)
    pattern = re.compile(r"^(refinement|padding)_(\d+)_vs_(\d+)$")
    series = {}
    for key, metrics in values.items():
        match = pattern.match(key)
        if match and isinstance(metrics, dict):
            series.setdefault(match.group(1), []).append((int(match.group(2)), int(match.group(3)), metrics))
    report = {}
    for kind, items in series.items():
        items.sort()
        for (a1, b1, m1), (a2, b2, m2) in zip(items[:-1], items[1:]):
            if b1 != a2 or b1*a2 == 0 or b1/a1 != b2/a2:
                continue
            ratio = b1/a1
            report[f"{kind}_{a1}_{b1}_{b2}"] = {
                metric: observed_order_from_differences(m1[metric], m2[metric], ratio)
                for metric in m1 if metric in m2 and isinstance(m1[metric], (int, float))
                and metric.endswith("relative_L2")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    print(json.dumps(from_record(args.record), indent=2))


if __name__ == "__main__":
    main()
