import numpy as np
from optlab import phase_element, detector_masks


def test_phase_element_modulus():
    h = np.random.default_rng(0).uniform(0, 2e-6, (16, 16))
    lam = 633e-9
    assert np.max(np.abs(np.abs(phase_element(h, lam, 1.5, 1.0)) - 1)) < 1e-12
    k = 0.05
    t = phase_element(h, lam, 1.5, 1.0, kappa=k)
    assert np.max(np.abs(np.abs(t) - np.exp(-2 * np.pi * k * h / lam))) < 1e-12


def test_masks_disjoint():
    n, px = 64, 1e-5
    regs = [dict(cx_m=-2e-4, cy_m=0, w_m=1e-4, h_m=1e-4),
            dict(cx_m=0, cy_m=0, w_m=1e-4, h_m=1e-4),
            dict(cx_m=2e-4, cy_m=1e-4, w_m=1e-4, h_m=1e-4)]
    m = detector_masks(n, px, regs)
    assert m.shape == (3, n, n)
    assert m.sum(axis=0).max() <= 1
    assert all(mm.sum() > 0 for mm in m)
