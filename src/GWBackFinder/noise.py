"""LISA instrumental noise in the TT and AA TDI channels.

Python/NumPy port of ``noise.jl`` from GWBackFinder.jl, following
https://arxiv.org/abs/2009.11845. All functions are vectorised: ``f``, ``P``
and ``A`` can be scalars or NumPy arrays that broadcast against each other.

``P`` is the amplitude of the interferometric (IMS / optical metrology) noise
and ``A`` the amplitude of the test-mass acceleration noise.
"""
import numpy as np

#: LISA arm length [m]
L_ARM = 2.5e9
#: speed of light [m/s]
C_LIGHT = 3e8
#: H0 / h [1/s] used to convert strain PSD into h^2 Omega
H0_OVER_H = 3.24e-18


def _x(f, L, c):
    return 2 * np.pi * f * L / c


def noise_Pims(f, P, c=C_LIGHT):
    return P**2 * (1e-12) ** 2 * (1 + (2e-3 / f) ** 4) * (2 * np.pi * f / c) ** 2


def noise_Pacc(f, A, c=C_LIGHT):
    return (A**2 * (1e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f * 1e3 / 8) ** 4)
            * (1 / (2 * np.pi * f)) ** 4 * (2 * np.pi * f / c) ** 2)


def R_tilde_AA(f, L=L_ARM, c=C_LIGHT):
    return 9 / 20 / (1 + 0.7 * _x(f, L, c) ** 2)


def R_tilde_TT(f, L=L_ARM, c=C_LIGHT):
    x = _x(f, L, c)
    return 9 / 20 * x**6 / (1.8e3 + 0.7 * x**8)


def R_AA(f, L=L_ARM, c=C_LIGHT):
    x = _x(f, L, c)
    return 16 * np.sin(x) ** 2 * x**2 * R_tilde_AA(f, L, c)


def R_TT(f, L=L_ARM, c=C_LIGHT):
    x = _x(f, L, c)
    return 16 * np.sin(x) ** 2 * x**2 * R_tilde_TT(f, L, c)


def N_AA(f, P, A, L=L_ARM, c=C_LIGHT):
    x = _x(f, L, c)
    cx = np.cos(x)
    return 8 * np.sin(x) ** 2 * (4 * (1 + cx + cx**2) * noise_Pacc(f, A, c)
                                 + (2 + cx) * noise_Pims(f, P, c))


def N_TT(f, P, A, L=L_ARM, c=C_LIGHT):
    x = _x(f, L, c)
    cx = np.cos(x)
    return 16 * np.sin(x) ** 2 * (2 * (1 - cx) ** 2 * noise_Pacc(f, A, c)
                                  + (1 - cx) * noise_Pims(f, P, c))


def Sn_AA(f, P, A, L=L_ARM, c=C_LIGHT):
    return N_AA(f, P, A, L, c) / R_AA(f, L, c)


def Sn_TT(f, P, A, L=L_ARM, c=C_LIGHT):
    return N_TT(f, P, A, L, c) / R_TT(f, L, c)


def Omega_noiseh2_AA(f, L=L_ARM, c=C_LIGHT, P=15.0, A=3.0):
    """h^2 Omega_noise in the AA channel.

    Argument order ``(f, L, c, P, A)`` is the same as in the Julia package.
    """
    return 4 * np.pi**2 * f**3 * Sn_AA(f, P, A, L, c) / (3 * H0_OVER_H**2)


def Omega_noiseh2_TT(f, L=L_ARM, c=C_LIGHT, P=15.0, A=3.0):
    """h^2 Omega_noise in the TT channel (argument order as in Julia)."""
    return 4 * np.pi**2 * f**3 * Sn_TT(f, P, A, L, c) / (3 * H0_OVER_H**2)


def noise_basis(f, channel="AA", L=L_ARM, c=C_LIGHT):
    """Return ``(u, v)`` such that ``Omega_noise(f, P, A) = A**2 * u + P**2 * v``.

    The noise is a linear combination of ``A**2`` and ``P**2`` with
    frequency-dependent coefficients, so precomputing them once makes repeated
    simulations on a fixed frequency grid much cheaper.
    """
    fn = {"AA": Omega_noiseh2_AA, "TT": Omega_noiseh2_TT}[channel]
    u = fn(f, L, c, 0.0, 1.0)
    v = fn(f, L, c, 1.0, 0.0)
    return u, v
