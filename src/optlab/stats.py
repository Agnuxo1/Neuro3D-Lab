"""Estadistica: intervalo de Wilson para una proporcion binomial."""
from __future__ import annotations

import math


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        raise ValueError("n must be > 0")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))
