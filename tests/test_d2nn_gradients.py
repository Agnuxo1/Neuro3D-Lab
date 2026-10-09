import numpy as np
from optlab import DiffractiveNetwork, detector_masks


def test_gradient_fd():
    n, px, lam = 32, 16e-6, 633e-9
    regs = [dict(cx_m=(i - 1) * 8 * px, cy_m=0.0, w_m=5 * px, h_m=5 * px) for i in range(3)]
    masks = detector_masks(n, px, regs)
    net = DiffractiveNetwork(n, px, lam, [3e-3, 4e-3, 5e-3], masks, scale=1.0, seed=5)
    rng = np.random.default_rng(7)
    u0 = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    label = 1
    loss, grad = net.loss_and_grad(u0, label)
    assert grad.shape == net.phases.shape
    # elementos con gradiente significativo (evita cocientes ruido/ruido)
    top = np.argsort(-np.abs(grad).ravel())[:200]
    picks = rng.choice(top, 5, replace=False)
    h = 1e-6
    errs = []
    for idx in picks:
        ix = np.unravel_index(idx, grad.shape)
        orig = net.phases[ix]
        net.phases[ix] = orig + h
        lp, _ = net.loss_and_grad(u0, label)
        net.phases[ix] = orig - h
        lm, _ = net.loss_and_grad(u0, label)
        net.phases[ix] = orig
        fd = (lp - lm) / (2 * h)
        errs.append(abs(fd - grad[ix]) / max(abs(fd), abs(grad[ix])))
    print("rel errs", errs)
    assert max(errs) < 1e-6, errs
