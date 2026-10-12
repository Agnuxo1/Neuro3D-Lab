"""Independent differentiable implementation and fixed digital comparison models."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import DiffractiveNetwork


class TorchD2NN(nn.Module):
    """Mirror the physical-support convention with independent Torch FFT/autograd."""
    def __init__(self, reference: DiffractiveNetwork):
        super().__init__()
        self.optics, self.beta = reference.optics, reference.beta
        self.phases = nn.Parameter(torch.tensor(reference.phases, dtype=torch.float64))
        self.register_buffer("kernels", torch.tensor(np.stack(reference.kernels), dtype=torch.complex128))
        self.register_buffer("masks", torch.tensor(reference.full_masks, dtype=torch.float64))
        self.register_buffer("aperture", torch.tensor(reference.aperture, dtype=torch.float64))
        self.offset = (self.optics.m-self.optics.n*self.optics.refinement)//2

    def forward(self, images):
        """Accept cell-energy amplitudes in the physical n x n support."""
        o, off = self.optics, self.offset
        if images.ndim != 3 or tuple(images.shape[1:]) != (o.n, o.n):
            raise ValueError("TorchD2NN requires (batch,n,n) inputs")
        r = o.refinement
        sampled = images.to(torch.complex128).repeat_interleave(r, -2).repeat_interleave(r, -1)/r
        a = F.pad(sampled, (off, off, off, off))
        fine_phases = self.phases.repeat_interleave(r, -2).repeat_interleave(r, -1)
        phase = F.pad(torch.exp(1j*fine_phases), (off, off, off, off),
                      value=0 if o.boundary == "aperture" else 1)
        for k in range(o.layers):
            a = torch.fft.ifft2(self.kernels[k]*torch.fft.fft2(a, norm="ortho"), norm="ortho")*phase[k]
        a = torch.fft.ifft2(self.kernels[-1]*torch.fft.fft2(a, norm="ortho"), norm="ortho")
        if o.boundary == "aperture":
            a = a*self.aperture
        intensity = a.real.square()+a.imag.square()
        return self.beta*torch.einsum("kij,bij->bk", self.masks, intensity)


class QuadraticReadout(nn.Module):
    """Parameter-matched positive quadratic scores on shared learned features."""
    def __init__(self, size=64, classes=10, rank=12):
        super().__init__()
        self.features = nn.Linear(size*size, rank, bias=False)
        self.readout = nn.Parameter(torch.randn(classes, rank)*0.1)

    def forward(self, inputs):
        """Compute nonnegative quadratic forms without calling an optical model."""
        features = self.features(inputs.flatten(1)).square()
        return features @ F.softplus(self.readout).transpose(0, 1)


def digital_model(name, size=64, classes=10):
    """Construct baselines with explicit architectures and parameter counts."""
    flat = size*size
    if name == "linear":
        model = nn.Sequential(nn.Flatten(), nn.Linear(flat, classes))
    elif name in {"mlp12", "mlp128"}:
        hidden = 12 if name == "mlp12" else 128
        model = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, classes))
    elif name == "quadratic12":
        model = QuadraticReadout(size, classes)
    elif name == "cnn":
        model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                              nn.Flatten(), nn.Linear(16*(size//4)**2, classes))
    else:
        raise ValueError("unknown baseline architecture")
    return model.to(dtype=torch.float64)


def check_equivalence(reference, inputs, labels, device="cpu"):
    """Compare scores, loss, and complete gradient without training or TEST."""
    model = TorchD2NN(reference).to(device)
    x = torch.as_tensor(inputs, dtype=torch.complex128, device=device)
    y = torch.as_tensor(labels, dtype=torch.int64, device=device)
    scores = model(x)
    loss = F.cross_entropy(scores, y)
    loss.backward()
    expected = reference.forward(inputs)["scores"]
    loss_np, grad_np = reference.loss_and_grad(inputs, labels)
    actual, gradient = scores.detach().cpu().numpy(), model.phases.grad.detach().cpu().numpy()
    score_error = float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected), 1e-30))
    grad_error = float(np.linalg.norm(gradient-grad_np)/max(np.linalg.norm(grad_np), 1e-30))
    loss_error = abs(float(loss.detach().cpu())-loss_np)
    values = dict(score_relative=score_error, loss_absolute=loss_error, gradient_relative=grad_error)
    values["pass"] = score_error <= 1e-9 and loss_error <= 1e-10 and grad_error <= 1e-8
    return values
