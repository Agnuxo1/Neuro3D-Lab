# SPDX-License-Identifier: Apache-2.0
"""Mach-Zehnder interferometer (2x2 matrices).

B = (1/sqrt(2)) [[1, i], [i, 1]] is an ideal 50/50 beam splitter, and
P(theta) = diag(exp(i*theta), 1) is a phase shifter on the first mode.
MZI(theta, phi) = B P(theta) B P(phi). B and P(theta) are unitary, so MZI is unitary.
For phi = 0, |T11|^2 = sin^2(theta/2), where T11 is the (1,1) element of MZI.
"""
from __future__ import annotations

import numpy as np


def beamsplitter():
    """Return the ideal 50/50 beam-splitter matrix B = (1/sqrt(2)) [[1, i], [i, 1]]."""
    return np.array([[1, 1j], [1j, 1]], dtype=complex) / np.sqrt(2)


def phase_shifter(theta):
    """Return the phase-shifter matrix P(theta) = diag(exp(i*theta), 1)."""
    return np.diag([np.exp(1j * theta), 1.0 + 0j])


def mzi(theta, phi):
    """Return the Mach-Zehnder matrix B P(theta) B P(phi)."""
    B = beamsplitter()
    return B @ phase_shifter(theta) @ B @ phase_shifter(phi)


def is_unitary(U, atol=1e-12):
    """Return True if max|U^H U - I| <= atol.

    The tolerance is absolute: there is no relative term, unlike np.allclose.
    U must be a square matrix.
    """
    U = np.asarray(U)
    # Absolute tolerance only (np.allclose would add rtol=1e-5).
    return bool(np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0]))) <= atol)
