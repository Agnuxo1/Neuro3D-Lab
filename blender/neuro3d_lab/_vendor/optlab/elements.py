"""Elementos opticos delgados y mascaras de detector.

Elemento de fase de espesor h en medio n_med, material n_mat + i*kappa:
t = exp(i*2*pi*(n_mat-n_med)*h/lambda) * exp(-2*pi*kappa*h/lambda).
Mascaras: coordenadas respecto al centro de la rejilla, x_j=(j-(n-1)/2)*pixel
(columna j -> +x hacia la derecha), y_i=-(i-(n-1)/2)*pixel (fila 0 = arriba =
+y maximo, como en la imagen de Blender); region semiabierta [c-w/2, c+w/2).
"""
from __future__ import annotations

import numpy as np


def phase_element(height_m, wavelength_m, n_material, n_medium, kappa=0.0):
    h = np.asarray(height_m, dtype=float)
    phase = 2.0 * np.pi * (n_material - n_medium) * h / wavelength_m
    return np.exp(1j * phase) * np.exp(-2.0 * np.pi * kappa * h / wavelength_m)


def detector_masks(n, pixel_m, regions):
    coords = (np.arange(n) - (n - 1) / 2.0) * pixel_m
    # +y points up: row 0 (top of the image) has the largest y.
    X, Y = np.meshgrid(coords, -coords, indexing="xy")
    masks = np.zeros((len(regions), n, n), dtype=float)
    for k, r in enumerate(regions):
        cx, cy, w, h = r["cx_m"], r["cy_m"], r["w_m"], r["h_m"]
        m = (X >= cx - w / 2) & (X < cx + w / 2) & (Y >= cy - h / 2) & (Y < cy + h / 2)
        masks[k] = m
    return masks
