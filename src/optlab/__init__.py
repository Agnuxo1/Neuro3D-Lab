"""optlab: laboratorio optico (campo escalar, SI, exp(-i w t))."""
from .units import wavenumber, phase_from_opl, optical_path_length
from .propagation import transfer_function, angular_spectrum, angular_spectrum_adjoint
from .elements import phase_element, detector_masks
from . import fresnel
from .fresnel import normal_incidence, oblique
from .interferometer import beamsplitter, phase_shifter, mzi, is_unitary
from .d2nn import DiffractiveNetwork
from .stats import wilson_interval
from .bench import load_bench, simulate_bench
from .registry import append_run

__version__ = "0.1.0"
__all__ = [
    "wavenumber", "phase_from_opl", "optical_path_length",
    "transfer_function", "angular_spectrum", "angular_spectrum_adjoint",
    "phase_element", "detector_masks", "fresnel", "normal_incidence", "oblique",
    "beamsplitter", "phase_shifter", "mzi", "is_unitary",
    "DiffractiveNetwork", "wilson_interval", "load_bench", "simulate_bench", "append_run",
]
