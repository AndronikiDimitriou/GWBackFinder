"""Frequency grid and coarse-graining (port of ``useful_functions.jl``).

Index conventions
-----------------
Julia is 1-based, Python is 0-based. Everything returned here is 0-based:

* ``idx[j]`` is the 1000-bin index (0..999) of ``f_filtered[j]``;
* ``idx_26[j]`` is the 26-bin index (0..25) of ``f[j]``.

The Julia constant ``2971`` (first coarse-grained point, f = 3e-3 Hz) becomes
``N_UNBINNED = 2970`` here: ``f[:2970]`` is kept at full resolution and
``f[2970:]`` is coarse-grained into 1000 log-spaced bins.
"""
import numpy as np

F_MIN = 3e-5
F_MAX = 0.5
DF = 1e-6
#: number of low-frequency points kept unbinned (Julia: ``f[1:2970]``)
N_UNBINNED = 2970
#: number of log-spaced coarse-graining bins above 3 mHz
N_BINS = 1000
#: number of log-spaced edges defining the piecewise power law (N_c = 27 in the
#: paper). The trained posterior_AA.pkl uses 27; the GWBackFinder.jl version on
#: GitHub has 26, which shifts every reconstruction by ~0.08 dex in frequency.
N_EDGES_26 = 27


def frequency_grid(f_min=F_MIN, f_max=F_MAX, df=DF):
    """Equivalent of Julia's ``range(start=3e-5, stop=0.5, step=1e-6)``.

    Values are computed as ``k / 1e6`` so each point is the correctly rounded
    decimal, like Julia's ``StepRangeLen`` (``np.arange`` with a float step
    accumulates rounding error).
    """
    k0 = round(f_min / df)
    k1 = round(f_max / df)
    return np.arange(k0, k1 + 1, dtype=np.float64) / round(1 / df)


def _searchsortedlast(edges, x):
    """0-based version of Julia's ``searchsortedlast`` (minus one)."""
    return np.searchsorted(edges, x, side="right") - 1


def binning(f, n_edges=N_EDGES_26):
    """Bin the frequency grid exactly as ``GWBackFinder.binning`` in Julia.

    Returns
    -------
    idx : (len(f) - 2970,) int array
        0-based index of the 1000-bin coarse-graining bin of each ``f_filtered``.
    idx_26 : (len(f),) int array
        0-based index of the 26-bin interval of each ``f``.
    logbins_26 : (n_edges,) array
        Log-spaced edges between 3e-5 and 0.5 Hz (27 by default, as used for the
        trained posteriors). ``Sb1 ... Sb25`` in the Julia code are
        ``logbins_26[0] ... logbins_26[24]`` and the pivot ``Sb13`` is ``logbins_26[12]``.
    logbins : (1000,) array
        Edges of the 1000 log-spaced bins between 3e-3 and 0.5 Hz.
    f_filtered : array
        ``f[2970:]``.
    """
    f = np.asarray(f, dtype=np.float64)
    f_filtered = f[N_UNBINNED:]
    logbins = 10 ** np.linspace(np.log10(f[N_UNBINNED]), np.log10(F_MAX), N_BINS)
    idx = _searchsortedlast(logbins, f_filtered)
    idx[0] = 0
    logbins_26 = 10 ** np.linspace(np.log10(F_MIN), np.log10(F_MAX), n_edges)
    idx_26 = _searchsortedlast(logbins_26, f)
    idx_26[0] = 0
    return idx, idx_26, logbins_26, logbins, f_filtered


def inverse_variance_bin(values, weights, idx, n_bins=N_BINS):
    """Inverse-noise weighted average of ``values`` in each bin.

    ``weights`` are ``1 / Omega_noise``. ``values`` may be 1-D or have a
    leading batch dimension. Empty bins give NaN (Julia would give 0 -> -Inf
    after the log; with the default grid no bin is empty).
    """
    norm = np.bincount(idx, weights=weights, minlength=n_bins)
    values = np.asarray(values)
    if values.ndim == 1:
        num = np.bincount(idx, weights=values * weights, minlength=n_bins)
    else:
        num = np.stack([np.bincount(idx, weights=v * weights, minlength=n_bins) for v in values])
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / norm
