"""Piecewise power-law signal model (port of ``f26`` in ``mock_signal.jl``).

The GW spectrum is a continuous broken power law in 26 log-spaced segments::

    segment 0 : f <= Sb1                     slope n0
    segment k : Sb_k < f <= Sb_{k+1}         slope n_k   (k = 1..24)
    segment 25: f > Sb25                     slope n25

normalised so that ``Omega(S0) = As`` with the pivot ``S0 = Sb13``.
In log space this is piecewise linear, which lets us evaluate the 26 Julia
``if/elseif`` branches as a single vectorised expression.
"""
import numpy as np

N_SLOPES = 26


def _offsets(slopes, log_breaks, pivot_segment):
    """Per-segment log offsets c_k so that log(Omega/As) = n_k * x + c_k."""
    dn = slopes[..., 1:] - slopes[..., :-1]               # (..., 25)
    term = dn * log_breaks                                 # boundary k sits at Sb_{k+1}
    S = np.concatenate([np.zeros(slopes.shape[:-1] + (1,)), np.cumsum(term, axis=-1)], axis=-1)
    return S[..., pivot_segment:pivot_segment + 1] - S  # (..., 26)


def piecewise_powerlaw(f, slopes, amplitude, breakpoints, f_pivot=None, log=False):
    """Evaluate the 26-segment broken power law.

    Parameters
    ----------
    f : array (n_f,)
        Frequencies.
    slopes : array (26,) or (B, 26)
        Slopes ``n0 ... n25``.
    amplitude : float or array (B,)
        Linear amplitude ``As`` at the pivot (``10**Amp``).
    breakpoints : array (25,)
        ``Sb1 ... Sb25`` (``logbins_26[:25]``).
    f_pivot : float, optional
        Pivot frequency ``S0``; defaults to ``Sb13 = breakpoints[12]``.
    log : bool
        If True return ``log(Omega)`` (natural log) instead of ``Omega``.

    Returns
    -------
    array (n_f,) or (B, n_f)
    """
    f = np.asarray(f, dtype=np.float64)
    breakpoints = np.asarray(breakpoints, dtype=np.float64)
    slopes = np.asarray(slopes, dtype=np.float64)
    if slopes.shape[-1] != N_SLOPES or breakpoints.shape != (N_SLOPES - 1,):
        raise ValueError("expected 26 slopes and 25 breakpoints")
    if f_pivot is None:
        f_pivot = breakpoints[12]
    seg = np.searchsorted(breakpoints, f, side="left")             # segment of each f
    pivot_segment = int(np.searchsorted(breakpoints, f_pivot, side="left"))
    x = np.log(f / f_pivot)
    c = _offsets(slopes, np.log(breakpoints / f_pivot), pivot_segment)
    amp = np.log(np.asarray(amplitude, dtype=np.float64))
    out = slopes[..., seg] * x + c[..., seg] + np.expand_dims(amp, -1)
    return out if log else np.exp(out)


def f26(S, S0, *args):
    """Drop-in for the Julia signature ``f26(S, S0, n0..n25, As, Sb1..Sb25)``."""
    if len(args) != 26 + 1 + 25:
        raise ValueError("f26 expects 26 slopes, As and 25 breakpoints")
    slopes = np.array(args[:26], dtype=np.float64)
    As = args[26]
    breaks = np.array(args[27:], dtype=np.float64)
    out = piecewise_powerlaw(np.atleast_1d(S), slopes, As, breaks, f_pivot=S0)
    return out[0] if np.ndim(S) == 0 else out


def theta_to_physical(theta):
    """Map unit-cube parameters to physical ones (as in ``model_train_data``).

    ``theta[..., :26]`` -> slopes in [-12, 12], ``theta[..., 26]`` -> log10
    amplitude in [-14, -6], ``theta[..., 27]`` is the noise parameter ``A``.
    """
    theta = np.asarray(theta, dtype=np.float64)
    slopes = -12 + theta[..., :26] * 24
    log10_amp = -14 + theta[..., 26] * 8
    A = theta[..., 27] if theta.shape[-1] > 27 else None
    return slopes, log10_amp, A
