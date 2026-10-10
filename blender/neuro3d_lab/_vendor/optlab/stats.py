# SPDX-License-Identifier: Apache-2.0
"""Statistics: Wilson score interval for a binomial proportion."""
from __future__ import annotations

import math


def wilson_interval(successes, n, z=1.96):
    """Return the approximate Wilson score confidence interval (lower, upper) for p = successes / n.

    The bounds are clipped to [0, 1]. z is the standard normal quantile; the default
    z = 1.96 gives a nominal two-sided level of 95%. The interval is approximate: its
    actual coverage is not exact, particularly for small n or for p near 0 or 1.
    n must be > 0. The caller must supply 0 <= successes <= n; this is not checked.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
