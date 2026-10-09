"""Interferometro Mach-Zehnder (matrices 2x2).

B = (1/sqrt2)[[1,i],[i,1]]; P(theta)=diag(e^{i theta},1);
MZI(theta,phi) = B P(theta) B P(phi). Con phi=0: |T11|^2 = sin^2(theta/2).
"""
from __future__ import annotations

import numpy as np


def beamsplitter():
    return np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)


def phase_shifter(theta):
    return np.diag([np.exp(1j * theta), 1.0 + 0j])


def mzi(theta, phi):
    B = beamsplitter()
    return B @ phase_shifter(theta) @ B @ phase_shifter(phi)


def is_unitary(U, atol=1e-12):
    U = np.asarray(U)
    # max|U^H U - I| <= atol, sin tolerancia relativa (np.allclose anade rtol=1e-5)
    return bool(np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0]))) <= atol)
