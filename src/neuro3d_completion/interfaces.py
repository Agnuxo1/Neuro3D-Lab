"""Validated Fresnel coefficients with nonzero evanescent transmission amplitudes."""
from __future__ import annotations

import cmath
import math
from .core import positive


def normal_incidence(n1, n2):
    """Return amplitudes and normal flux ratios for positive lossless indices."""
    a, b = positive(n1, "n1"), positive(n2, "n2")
    r, t = (a-b)/(a+b), 2*a/(a+b)
    return dict(r=r, t=t, R=r*r, T=(b/a)*t*t)


def oblique(n1, n2, theta_i_rad, polarization):
    """Retain the existing p-reflection sign convention and validate incidence."""
    a, b = positive(n1, "n1"), positive(n2, "n2")
    angle = float(theta_i_rad)
    if not math.isfinite(angle) or angle < 0 or angle >= math.pi/2:
        raise ValueError("theta_i_rad must lie in [0, pi/2)")
    if polarization not in {"s", "p"}:
        raise ValueError("polarization must be 's' or 'p'")
    ci, st = math.cos(angle), a*math.sin(angle)/b
    ct = cmath.sqrt(1-st*st)
    if polarization == "s":
        denominator = a*ci+b*ct
        r, t = (a*ci-b*ct)/denominator, 2*a*ci/denominator
    else:
        denominator = a*ct+b*ci
        r, t = (a*ct-b*ci)/denominator, 2*a*ci/denominator
    reflectance = float(abs(r)**2)
    transmittance = 0.0 if st >= 1 else float(b*ct.real/(a*ci)*abs(t)**2)
    return dict(r=r, t=t, R=reflectance, T=transmittance)
