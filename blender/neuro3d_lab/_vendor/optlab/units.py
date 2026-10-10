# SPDX-License-Identifier: Apache-2.0
"""Units and optical phase (SI: meters, seconds).

Convention: time dependence exp(-i*w*t). In a medium of refractive index n, the
wavenumber is k = 2*pi*n/lambda_0, where lambda_0 is the vacuum wavelength (the
wavelength inside the medium is lambda_0/n). The phase of an optical path is
phi = 2*pi*OPL/lambda_0, with OPL = sum n_i*d_i.
"""
from __future__ import annotations

import math


def wavenumber(wavelength_m: float, n: float = 1.0) -> float:
    """Return the wavenumber k = 2*pi*n/wavelength_m in a medium of index n.

    wavelength_m is the vacuum wavelength in meters. The default n = 1.0 gives vacuum.
    """
    return 2.0 * math.pi * n / wavelength_m


def optical_path_length(segments) -> float:
    """Return the optical path length OPL = sum of n_i * d_i.

    segments is an iterable of (n_i, d_i) pairs: the refractive index n_i and the
    geometric length d_i in meters of each segment.
    """
    return float(sum(n * d for n, d in segments))


def phase_from_opl(opl_m: float, wavelength_m: float) -> float:
    """Return the phase phi = 2*pi*OPL/wavelength_m for an optical path length opl_m.

    opl_m and wavelength_m are in meters; wavelength_m is the vacuum wavelength.
    """
    return 2.0 * math.pi * opl_m / wavelength_m
