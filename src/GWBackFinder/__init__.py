"""GWBackFinder: simulation-based reconstruction of gravitational-wave backgrounds.

Pure-Python version of https://github.com/AndronikiDimitriou/GWBackFinder.jl
plus the sbi training code, for https://arxiv.org/abs/2309.08430.

The simulator only needs NumPy; ``prior``, ``train`` and ``posterior`` need
``torch`` and ``sbi`` and are imported lazily.
"""
from .binning import N_BINS, N_UNBINNED, binning, frequency_grid
from .noise import Omega_noiseh2_AA, Omega_noiseh2_TT
from .signal import f26, piecewise_powerlaw
from .simulator import Simulator
from .templates import TEMPLATES, template_omega

__all__ = [
    "Simulator", "binning", "frequency_grid", "piecewise_powerlaw", "f26",
    "Omega_noiseh2_AA", "Omega_noiseh2_TT", "template_omega", "TEMPLATES",
    "N_BINS", "N_UNBINNED",
]
__version__ = "1.0.0"
