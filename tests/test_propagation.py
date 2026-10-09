import numpy as np
from optlab import angular_spectrum, angular_spectrum_adjoint

LAM = 633e-9


def gauss(n, px, w):
    x = (np.arange(n) - n / 2) * px
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2 + Y**2) / w**2).astype(complex)


def test_uniform_field_phase():
    n, px, d = 64, 1e-6, 1.234e-3
    out = angular_spectrum(np.ones((n, n), complex), px, LAM, d)
    assert np.max(np.abs(out - np.exp(1j * 2 * np.pi * d / LAM))) < 1e-12


def test_energy_conservation():
    n, px = 256, LAM / 4
    u = gauss(n, px, 20 * px)
    out = angular_spectrum(u, px, LAM, 0.002)
    e0, e1 = np.sum(np.abs(u) ** 2), np.sum(np.abs(out) ** 2)
    assert abs(e1 - e0) / e0 < 1e-10


def test_inversion():
    n, px = 128, 1e-6
    u = gauss(n, px, 10e-6)
    back = angular_spectrum(angular_spectrum(u, px, LAM, 1e-3), px, LAM, -1e-3)
    assert np.max(np.abs(back - u)) < 1e-10


def test_adjoint():
    rng = np.random.default_rng(1)
    n, px, d = 32, 2e-6, 3e-4
    u = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    v = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    lhs = np.vdot(v, angular_spectrum(u, px, LAM, d))  # <Pu, v>
    rhs = np.vdot(angular_spectrum_adjoint(v, px, LAM, d), u)  # <u, P^H v>
    assert abs(lhs - rhs) / abs(lhs) < 1e-10
