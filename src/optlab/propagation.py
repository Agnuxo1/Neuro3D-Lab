"""Propagacion libre por espectro angular (campo escalar, exp(-i w t)).

H(fx,fy) = exp(i*2*pi*d*sqrt(1/lambda^2 - fx^2 - fy^2)) si fx^2+fy^2 < 1/lambda^2,
0 si es evanescente. fx,fy = np.fft.fftfreq(n, d=pixel_m). P_d(u) = ifft2(H*fft2(u)).
Adjunto (convencion dL = 2 Re(conj(g) dz), g = dL/dz*): P_d^H(g) = ifft2(conj(H)*fft2(g)).
"""
from __future__ import annotations

import numpy as np


def transfer_function(n: int, pixel_m: float, wavelength_m: float, distance_m: float) -> np.ndarray:
    f = np.fft.fftfreq(n, d=pixel_m)
    fx, fy = np.meshgrid(f, f, indexing="xy")
    arg = 1.0 / wavelength_m**2 - fx**2 - fy**2
    prop = arg > 0
    kz = np.sqrt(np.where(prop, arg, 0.0))
    return np.where(prop, np.exp(1j * 2.0 * np.pi * distance_m * kz), 0.0 + 0.0j)


def angular_spectrum(u, pixel_m, wavelength_m, distance_m):
    u = np.asarray(u, dtype=complex)
    H = transfer_function(u.shape[-1], pixel_m, wavelength_m, distance_m)
    return np.fft.ifft2(H * np.fft.fft2(u))


def angular_spectrum_adjoint(g, pixel_m, wavelength_m, distance_m):
    g = np.asarray(g, dtype=complex)
    H = transfer_function(g.shape[-1], pixel_m, wavelength_m, distance_m)
    return np.fft.ifft2(np.conj(H) * np.fft.fft2(g))
