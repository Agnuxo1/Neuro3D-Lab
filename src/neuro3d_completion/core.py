"""Finite-window scalar optics with fixed physical phase support and exact adjoints.

Arrays represent cell-energy amplitudes: their squared Euclidean norm is energy.
Increasing padding never adds trainable phase pixels. The enlarged FFT window
is still periodic; its adequacy must be established by a separate convergence run.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def positive(value, name):
    """Return a finite positive float or reject an invalid physical parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class Optics:
    """Separate physical support, numerical window, and boundary convention."""
    n: int = 128
    pixel_m: float = 0.0004
    wavelength_m: float = 0.00075
    distances_m: tuple = (0.018, 0.018, 0.018, 0.018)
    padding: int = 2
    refinement: int = 1
    boundary: str = "aperture"
    filter_mode: str = "none"

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 4 or self.n % 2:
            raise ValueError("n must be an even integer >= 4")
        if isinstance(self.padding, bool) or not isinstance(self.padding, int) or self.padding < 1:
            raise ValueError("padding must be a positive integer")
        if isinstance(self.refinement, bool) or not isinstance(self.refinement, int) or self.refinement < 1:
            raise ValueError("refinement must be a positive integer")
        positive(self.pixel_m, "pixel_m")
        positive(self.wavelength_m, "wavelength_m")
        if len(self.distances_m) < 1:
            raise ValueError("at least one propagation distance is required")
        for d in self.distances_m:
            positive(d, "distance_m")
        if self.boundary not in {"periodic", "transparent", "aperture"}:
            raise ValueError("unknown boundary convention")
        if self.boundary == "periodic" and self.padding != 1:
            raise ValueError("the legacy periodic convention requires padding=1")
        if self.boundary != "periodic" and self.padding < 2:
            raise ValueError("finite-support conventions require padding >= 2")
        if self.filter_mode not in {"none", "phase_gradient"}:
            raise ValueError("unknown spectral filter")

    @property
    def m(self):
        return self.n * self.refinement * self.padding

    @property
    def sample_pitch_m(self):
        return self.pixel_m / self.refinement

    @property
    def layers(self):
        return len(self.distances_m) - 1


def detector_masks(n, pixel_m, regions=None):
    """Build half-open detector regions with +x right and +y up."""
    if regions is None:
        regions = [dict(cx_m=c * 0.0004, cy_m=r * 0.0004,
                        w_m=0.0048, h_m=0.0048)
                   for r in (20, -20) for c in (-40, -20, 0, 20, 40)]
    x = (np.arange(n) - (n - 1) / 2) * positive(pixel_m, "pixel_m")
    xx, yy = np.meshgrid(x, -x, indexing="xy")
    masks = []
    for region in regions:
        cx, cy = float(region["cx_m"]), float(region["cy_m"])
        w, h = positive(region["w_m"], "w_m"), positive(region["h_m"], "h_m")
        if not np.isfinite([cx, cy]).all():
            raise ValueError("detector centers must be finite")
        masks.append((xx >= cx - w/2) & (xx < cx + w/2) &
                     (yy >= cy - h/2) & (yy < cy + h/2))
    if not masks:
        raise ValueError("at least one detector is required")
    result = np.asarray(masks, dtype=float)
    if np.any(result.sum(axis=(1, 2)) == 0):
        raise ValueError("every detector must contain at least one pixel")
    if np.any(result.sum(axis=0) > 1):
        raise ValueError("detector regions must be disjoint")
    return result


class DiffractiveNetwork:
    """Batched NumPy network with a fixed number of physical phase parameters."""
    def __init__(self, optics: Optics, masks, beta=100.0, seed=0):
        self.optics = optics
        self.beta = positive(beta, "beta")
        self.masks = np.asarray(masks, dtype=float)
        if (self.masks.ndim != 3 or self.masks.shape[1:] != (optics.n, optics.n)
                or len(self.masks) < 2 or not np.isfinite(self.masks).all()
                or np.any(self.masks < 0) or np.any(self.masks > 1)):
            raise ValueError("masks must be finite (classes,n,n) values in [0,1]")
        self.phases = np.random.default_rng(seed).uniform(
            0, 2*np.pi, (optics.layers, optics.n, optics.n))
        active = optics.n * optics.refinement
        off = (optics.m - active) // 2
        self.roi = (slice(off, off + active), slice(off, off + active))
        self.aperture = np.zeros((optics.m, optics.m), dtype=float)
        self.aperture[self.roi] = 1
        self.full_masks = np.zeros((len(masks), optics.m, optics.m))
        self.full_masks[(slice(None), *self.roi)] = np.repeat(
            np.repeat(self.masks, optics.refinement, axis=1), optics.refinement, axis=2)
        self.kernels = [self._kernel(d) for d in optics.distances_m]

    def _kernel(self, distance):
        """Build a propagating-only transfer function; filtering is explicit."""
        o = self.optics
        freq = np.fft.fftfreq(o.m, d=o.sample_pitch_m)
        fx, fy = np.meshgrid(freq, freq, indexing="xy")
        q2 = o.wavelength_m**-2 - fx**2 - fy**2
        propagating = q2 > 0
        q = np.sqrt(np.maximum(q2, 0))
        keep = propagating.copy()
        if o.filter_mode == "phase_gradient":
            window = o.m * o.sample_pitch_m
            keep &= (2*distance*np.abs(fx) <= window*q)
            keep &= (2*distance*np.abs(fy) <= window*q)
        return np.where(keep, np.exp(2j*np.pi*distance*q), 0j)

    @staticmethod
    def propagate(field, kernel, adjoint=False):
        """Apply the FFT operator or its Hermitian adjoint on the last axes."""
        multiplier = np.conj(kernel) if adjoint else kernel
        return np.fft.ifft2(multiplier*np.fft.fft2(field, norm="ortho"), norm="ortho")

    def transmissions(self):
        """Embed physical phases in a transparent or absorbing exterior."""
        o = self.optics
        if self.phases.shape != (o.layers, o.n, o.n) or not np.isfinite(self.phases).all():
            raise ValueError("invalid phase tensor")
        outside = 0 if o.boundary == "aperture" else 1
        t = np.full((o.layers, o.m, o.m), outside, dtype=complex)
        phase = np.repeat(np.repeat(self.phases, o.refinement, axis=1), o.refinement, axis=2)
        t[(slice(None), *self.roi)] = np.exp(1j*phase)
        return t

    def forward(self, inputs, cache=False):
        """Return intensities and scores, preserving a leading batch axis."""
        u = np.asarray(inputs, dtype=complex)
        single = u.ndim == 2
        if single:
            u = u[None]
        if u.ndim != 3 or u.shape[1:] != (self.optics.n, self.optics.n) or len(u) == 0:
            raise ValueError("inputs must have shape (n,n) or (batch,n,n)")
        if not np.isfinite(u).all():
            raise ValueError("inputs must be finite")
        a = np.zeros((len(u), self.optics.m, self.optics.m), dtype=complex)
        r = self.optics.refinement
        a[(slice(None), *self.roi)] = np.repeat(np.repeat(u, r, axis=1), r, axis=2)/r
        t = self.transmissions()
        states = []
        for k in range(self.optics.layers):
            a = self.propagate(a, self.kernels[k]) * t[k]
            if cache:
                states.append(a)
        f = self.propagate(a, self.kernels[-1])
        if self.optics.boundary == "aperture":
            f = f * self.aperture
        intensity = np.abs(f)**2
        scores = self.beta*np.einsum("kij,bij->bk", self.full_masks, intensity)
        if cache:
            return dict(I=intensity, scores=scores, f=f, states=states, t=t)
        return dict(I=intensity[0] if single else intensity,
                    scores=scores[0] if single else scores,
                    f=f[0] if single else f)

    def loss_and_grad(self, inputs, labels):
        """Compute mean cross-entropy and its complete analytic phase gradient."""
        c = self.forward(inputs, cache=True)
        y = np.asarray(labels)
        if y.ndim == 0:
            y = y[None]
        if (y.shape != (len(c["scores"]),) or not np.issubdtype(y.dtype, np.integer)
                or np.any(y < 0) or np.any(y >= len(self.masks))):
            raise ValueError("labels must be one valid integer per input")
        scores = c["scores"]
        maximum = np.max(scores, axis=1)
        normalizer = maximum + np.log(np.sum(np.exp(scores-maximum[:, None]), axis=1))
        loss = float(np.mean(normalizer - scores[np.arange(len(y)), y]))
        delta = np.exp(scores - normalizer[:, None])
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        g_intensity = self.beta*np.einsum("bk,kij->bij", delta, self.full_masks)
        g = g_intensity*c["f"]
        if self.optics.boundary == "aperture":
            g *= self.aperture
        g = self.propagate(g, self.kernels[-1], adjoint=True)
        gradient = np.empty_like(self.phases)
        for k in reversed(range(self.optics.layers)):
            full_grad = -2*np.imag(np.conj(g)*c["states"][k])
            r, n = self.optics.refinement, self.optics.n
            physical_grad = full_grad[(slice(None), *self.roi)].reshape(len(y), n, r, n, r)
            gradient[k] = physical_grad.sum(axis=(0, 2, 4))
            g = self.propagate(g*np.conj(c["t"][k]), self.kernels[k], adjoint=True)
        return loss, gradient

    def predict(self, inputs, batch_size=32):
        """Predict classes without allocating the complete dataset's fields."""
        u = np.asarray(inputs)
        if u.ndim == 2:
            return int(np.argmax(self.forward(u)["scores"]))
        if not isinstance(batch_size, int) or batch_size < 1 or u.ndim != 3 or len(u) == 0:
            raise ValueError("a nonempty batch and positive batch_size are required")
        return np.concatenate([np.argmax(self.forward(u[i:i+batch_size])["scores"], axis=1)
                               for i in range(0, len(u), batch_size)])

    def fit(self, inputs, labels, epochs=30, lr=0.02, batch_size=32, seed=0):
        """Train with minibatch Adam; return loss per epoch without using TEST."""
        x, y = np.asarray(inputs), np.asarray(labels)
        if x.ndim != 3 or y.shape != (len(x),) or len(x) == 0:
            raise ValueError("nonempty paired training inputs and labels are required")
        if (not isinstance(epochs, int) or epochs < 1 or
                not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("epochs and batch_size must be positive integers")
        positive(lr, "lr")
        if self.optics.layers == 0:
            raise ValueError("a network without phase layers has no trainable parameters")
        rng = np.random.default_rng(seed + 1000)
        first, second = np.zeros_like(self.phases), np.zeros_like(self.phases)
        history, step = [], 0
        for _ in range(epochs):
            total = 0
            indices = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                ix = indices[start:start+batch_size]
                loss, grad = self.loss_and_grad(x[ix], y[ix])
                if not np.isfinite(grad).all() or not math.isfinite(loss):
                    raise FloatingPointError("nonfinite loss or gradient")
                step += 1
                first = 0.9*first + 0.1*grad
                second = 0.999*second + 0.001*grad**2
                self.phases -= lr*(first/(1-0.9**step))/(np.sqrt(second/(1-0.999**step))+1e-8)
                total += loss*len(ix)
            history.append(total/len(y))
        return history


def score_error_bound(reference_field, field_error, masks, beta=1):
    """Bound detector score errors from a supplied valid L2 field-error bound."""
    f = np.asarray(reference_field, dtype=complex)
    e = float(field_error)
    masks = np.asarray(masks, dtype=float)
    if e < 0 or not math.isfinite(e) or not np.isfinite(f).all():
        raise ValueError("a finite field and nonnegative finite error bound are required")
    if (masks.ndim != 3 or masks.shape[1:] != f.shape or not np.isfinite(masks).all()
            or np.any(masks < 0) or np.any(masks > 1)):
        raise ValueError("masks must be (classes,*field.shape) weights in [0,1]")
    return positive(beta, "beta")*(2*np.linalg.norm(f)*e + e*e)


def certified_label(scores, score_error):
    """Return a robust argmax only when its margin exceeds twice the score bound."""
    s = np.asarray(scores, dtype=float)
    error = float(score_error)
    if s.ndim != 1 or len(s) < 2 or not np.isfinite(s).all() or error < 0 or not math.isfinite(error):
        raise ValueError("finite scores and a finite nonnegative error are required")
    order = np.argsort(s)
    margin = float(s[order[-1]] - s[order[-2]])
    return dict(label=int(order[-1]) if margin > 2*error else None,
                margin=margin, certified=bool(margin > 2*error))
