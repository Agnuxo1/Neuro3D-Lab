"""Unidades y fase optica (SI: metros, segundos).

Convencion: dependencia temporal exp(-i w t); k = 2*pi/lambda en el medio de
indice n; fase de camino optico phi = 2*pi*OPL/lambda con OPL = sum n_i*d_i.
"""
from __future__ import annotations

import math


def wavenumber(wavelength_m: float, n: float = 1.0) -> float:
    """Numero de onda en el medio de indice n: k = 2*pi*n/lambda_0."""
    return 2.0 * math.pi * n / wavelength_m


def optical_path_length(segments) -> float:
    """OPL = sum n_i * d_i para segments = iterable de (n, d_m)."""
    return float(sum(n * d for n, d in segments))


def phase_from_opl(opl_m: float, wavelength_m: float) -> float:
    """Fase de camino optico phi = 2*pi*OPL/lambda_0."""
    return 2.0 * math.pi * opl_m / wavelength_m
