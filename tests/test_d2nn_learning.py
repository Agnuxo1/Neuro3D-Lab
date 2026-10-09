import numpy as np
from optlab import DiffractiveNetwork, detector_masks


def make_data(n, px, per_class=10, seed=0):
    rng = np.random.default_rng(seed)
    x = (np.arange(n) - (n - 1) / 2) * px
    X, Y = np.meshgrid(x, x)
    centers = [(-6 * px, 0.0), (0.0, 0.0), (6 * px, 0.0)]
    U, y = [], []
    for c, (cx, cy) in enumerate(centers):
        for _ in range(per_class):
            amp = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (2 * px) ** 2))
            amp = amp + 0.02 * rng.normal(size=amp.shape)
            U.append(amp.astype(complex))
            y.append(c)
    return U, np.array(y)


def test_learning():
    n, px, lam = 32, 16e-6, 633e-9
    regs = [dict(cx_m=(i - 1) * 8 * px, cy_m=0.0, w_m=6 * px, h_m=6 * px) for i in range(3)]
    masks = detector_masks(n, px, regs)
    net = DiffractiveNetwork(n, px, lam, [3e-3, 3e-3, 3e-3], masks, scale=0.5, seed=0)
    U, y = make_data(n, px)
    hist = net.fit(U, y, epochs=60, lr=0.1)
    acc = np.mean([net.predict(u) == c for u, c in zip(U, y)])
    print("loss0", hist[0], "lossN", hist[-1], "acc", acc)
    assert hist[-1] < 0.5 * hist[0], (hist[0], hist[-1])
    assert acc >= 0.9, acc
