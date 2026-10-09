"""Red neuronal difractiva (D2NN) con gradiente analitico.

Campo escalar, exp(-i w t). a_0=u0; b_k=P_{d[k-1]}(a_{k-1}); a_k=b_k*exp(i phi_k)
(k=1..L); f=P_{d[L]}(a_L); I=|f|^2; s_c=scale*sum(mask_c*I); L=-log softmax(s)_y.
Retropropagacion con g=dL/dz* (dL = 2 Re(conj(g) dz)):
  g_f = gI*f ; g_{a_L} = P^H g_f ;  dL/dphi_k = -2 Im(conj(g_{a_k}) a_k) ;
  g_{b_k} = conj(exp(i phi_k)) g_{a_k} ; g_{a_{k-1}} = P^H g_{b_k}.
(Nota: el factor es conj(g)*a; sin conj el signo seria erroneo.)
"""
from __future__ import annotations

import numpy as np

from .propagation import angular_spectrum, angular_spectrum_adjoint


def _softmax(s):
    e = np.exp(s - np.max(s))
    return e / e.sum()


class DiffractiveNetwork:
    def __init__(self, n, pixel_m, wavelength_m, distances_m, detector_masks, scale=1.0, seed=0):
        self.n = int(n)
        self.pixel_m = float(pixel_m)
        self.wavelength_m = float(wavelength_m)
        self.distances_m = [float(d) for d in distances_m]
        self.L = len(self.distances_m) - 1
        if self.L < 1:
            raise ValueError("distances_m needs at least 2 elements (L>=1)")
        self.masks = np.asarray(detector_masks, dtype=float)
        if self.masks.ndim != 3 or self.masks.shape[1:] != (self.n, self.n):
            raise ValueError("detector_masks must have shape (K,n,n)")
        self.scale = float(scale)
        rng = np.random.default_rng(seed)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=(self.L, self.n, self.n))

    def _P(self, u, d):
        return angular_spectrum(u, self.pixel_m, self.wavelength_m, d)

    def _PH(self, g, d):
        return angular_spectrum_adjoint(g, self.pixel_m, self.wavelength_m, d)

    def forward(self, u0):
        a = np.asarray(u0, dtype=complex)
        a_list = [a]
        for k in range(self.L):
            b = self._P(a, self.distances_m[k])
            a = b * np.exp(1j * self.phases[k])
            a_list.append(a)
        f = self._P(a, self.distances_m[self.L])
        I = np.abs(f) ** 2
        scores = self.scale * np.tensordot(self.masks, I, axes=([1, 2], [0, 1]))
        return {"I": I, "scores": scores, "f": f, "a": a_list}

    def predict(self, u0):
        return int(np.argmax(self.forward(u0)["scores"]))

    def loss_and_grad(self, u0, label):
        c = self.forward(u0)
        p = _softmax(c["scores"])
        loss = float(-np.log(max(p[label], 1e-300)))
        delta = p.copy()
        delta[label] -= 1.0
        gI = self.scale * np.tensordot(delta, self.masks, axes=(0, 0))
        g = self._PH(gI * c["f"], self.distances_m[self.L])  # g_{a_L}
        grad = np.empty_like(self.phases)
        for k in range(self.L, 0, -1):
            a_k = c["a"][k]
            grad[k - 1] = -2.0 * np.imag(np.conj(g) * a_k)
            g_b = g * np.conj(np.exp(1j * self.phases[k - 1]))
            g = self._PH(g_b, self.distances_m[k - 1])
        return loss, grad

    def fit(self, U0_batch, labels, epochs, lr, betas=(0.9, 0.999), eps=1e-8):
        m = np.zeros_like(self.phases)
        v = np.zeros_like(self.phases)
        history = []
        N = len(labels)
        for t in range(1, int(epochs) + 1):
            tot = 0.0
            grad = np.zeros_like(self.phases)
            for u0, y in zip(U0_batch, labels):
                l, g = self.loss_and_grad(u0, int(y))
                tot += l
                grad += g
            grad /= N
            history.append(tot / N)
            m = betas[0] * m + (1 - betas[0]) * grad
            v = betas[1] * v + (1 - betas[1]) * grad**2
            mh = m / (1 - betas[0] ** t)
            vh = v / (1 - betas[1] ** t)
            self.phases = self.phases - lr * mh / (np.sqrt(vh) + eps)
        return history
