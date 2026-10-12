"""Design-stage power, minimum detectable effect and equivalence tests for paired classifiers.

Exact power of the two-sided McNemar test (conditional exact binomial, p = 1/2), as
used by evidence.paired_comparison, for n paired TEST samples with discordance
probability pd = p10 + p01 and accuracy difference delta = p10 - p01. Run this before
TEST is opened: a non-significant result is only informative about effects the design
could detect. Reference for McNemar in classifier comparison: Dietterich (1998),
Neural Computation 10(7), 1895-1923, DOI 10.1162/089976698300017197. TOST: Schuirmann
(1987), J. Pharmacokinet. Biopharm. 15, 657-680, DOI 10.1007/BF01068419.
"""
from __future__ import annotations

import argparse
import json
import math
import numpy as np
from scipy.stats import binom, norm


def _lower_critical(discordant, alpha):
    """Largest c with 2*P(B <= c) <= alpha for B ~ Bin(D, 1/2); -1 if none."""
    c = int(binom.ppf(alpha/2, discordant, 0.5))
    while c >= 0 and 2*binom.cdf(c, discordant, 0.5) > alpha:
        c -= 1
    while 2*binom.cdf(c + 1, discordant, 0.5) <= alpha:
        c += 1
    return c


def mcnemar_power(n, discordance, delta, alpha=0.05):
    """Exact power of the two-sided exact McNemar test (rejection in either direction)."""
    pd, d = float(discordance), float(delta)
    if not isinstance(n, int) or n < 1 or not 0 < pd <= 1 or abs(d) > pd or not 0 < alpha < 1:
        raise ValueError("require integer n >= 1, 0 < discordance <= 1, |delta| <= discordance")
    q = (pd + d)/(2*pd)  # P(first correct | discordant)
    lo = max(0, int(n*pd - 8*math.sqrt(n*pd*(1 - pd)) - 10))
    hi = min(n, int(n*pd + 8*math.sqrt(n*pd*(1 - pd)) + 10))
    power = 0.0
    for discordant in range(max(lo, 1), hi + 1):
        weight = binom.pmf(discordant, n, pd)
        c = _lower_critical(discordant, alpha)
        if c < 0:
            continue
        power += weight*(binom.cdf(c, discordant, q) + binom.sf(discordant - c - 1, discordant, q))
    return float(power)


def minimum_detectable_effect(n, discordance, alpha=0.05, power=0.8, tolerance=1e-5):
    """Smallest |delta| reaching the requested power (bisection); None if unattainable."""
    if mcnemar_power(n, discordance, discordance, alpha) < power:
        return None
    low, high = 0.0, float(discordance)
    while high - low > tolerance:
        mid = (low + high)/2
        if mcnemar_power(n, discordance, mid, alpha) >= power:
            high = mid
        else:
            low = mid
    return high


def wilson_halfwidth(accuracy, n, z=1.959963984540054):
    """Half-width of the 95% Wilson interval for one model on n independent samples."""
    p = float(accuracy)
    if not 0 <= p <= 1 or n < 1:
        raise ValueError("invalid accuracy or n")
    return float(z*math.sqrt(p*(1 - p)/n + z*z/(4*n*n))/(1 + z*z/n))


def paired_tost(y, first, second, margin, alpha=0.05):
    """Two one-sided tests for |acc(first) - acc(second)| < margin (normal approximation).

    Uses the paired-difference variance (pd - delta^2)/n. Equivalence is concluded only
    if both one-sided p-values are below alpha. Report it as a pre-registered analysis.
    """
    y, a, b = map(np.asarray, (y, first, second))
    if y.ndim != 1 or len(y) < 2 or a.shape != y.shape or b.shape != y.shape:
        raise ValueError("paired nonempty prediction vectors are required")
    if not 0 < margin < 1:
        raise ValueError("margin must lie in (0, 1)")
    x = (a == y).astype(float) - (b == y).astype(float)
    n, delta = len(x), float(x.mean())
    se = math.sqrt(max(float(np.mean(x*x)) - delta*delta, 1e-15)/n)
    p_lower, p_upper = float(norm.sf((delta + margin)/se)), float(norm.cdf((delta - margin)/se))
    return dict(difference=delta, standard_error=se, margin=margin, p_lower=p_lower, p_upper=p_upper,
                equivalent=bool(max(p_lower, p_upper) < alpha))


def design_table(n=10000, discordances=(0.02, 0.04, 0.06, 0.08, 0.10), alpha=0.05, power=0.8, family=5):
    """MDE for the uncorrected and the worst-case Holm (alpha/family) level."""
    rows = []
    for pd in discordances:
        rows.append(dict(discordance=pd,
                         mde=minimum_detectable_effect(n, pd, alpha, power),
                         mde_holm_worst_case=minimum_detectable_effect(n, pd, alpha/family, power)))
    return dict(n=n, alpha=alpha, power=power, family=family, rows=rows,
                accuracy_halfwidth_at_0_95=wilson_halfwidth(0.95, n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--discordances", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--family", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(design_table(args.n, tuple(args.discordances), args.alpha, args.power, args.family), indent=2))


if __name__ == "__main__":
    main()
