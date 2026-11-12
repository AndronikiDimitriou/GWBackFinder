"""Loading trained posteriors and reconstructing the GW spectrum.

Replaces the ``juliacall`` parts of ``examples/Plots.py``.
"""
import io
import pickle

import numpy as np

from .signal import piecewise_powerlaw

# Pickles made from the old repo layout refer to ``src.GWBackFinder.prior``.
_MODULE_RENAMES = {
    "src.GWBackFinder.prior": "GWBackFinder.prior",
    "src.GWBackFinder.train": "GWBackFinder.train",
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        module = _MODULE_RENAMES.get(module, module)
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            # allow posteriors trained on a GPU to be opened on a CPU-only machine
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_posterior(path):
    """Load a pickled sbi posterior/inference object (e.g. ``posterior_AA.pkl``).

    Works for pickles created with the old ``src.GWBackFinder`` import path and
    for GPU-trained posteriors on CPU-only machines. Only open files you trust:
    unpickling can execute code.
    """
    with open(path, "rb") as fh:
        return _Unpickler(fh).load()


def _flow_sample_in_box(posterior, contexts, n_dims_box, per_try=16, max_tries=200):
    """One sample per context straight from the flow, redrawing samples that fall
    outside the unit box of the first ``n_dims_box`` parameters.

    Same distribution as sbi's rejection sampling (the flow truncated to the
    prior support), but batched over all contexts, so much faster. Contexts for
    which no in-box sample is found are dropped (with a warning).
    """
    import warnings
    import torch

    flow = posterior.posterior_estimator
    n = contexts.shape[0]
    res, todo = None, torch.arange(n)
    with torch.no_grad():
        for _ in range(max_tries):
            s = flow.sample(per_try, context=contexts[todo])          # (len(todo), per_try, d)
            if res is None:
                res = torch.full((n, s.shape[-1]), float("nan"))
            ok = ((s[..., :n_dims_box] >= 0) & (s[..., :n_dims_box] <= 1)).all(-1)
            has = ok.any(1)
            first = ok.float().argmax(1)
            rows = torch.arange(len(todo))[has]
            res[todo[has]] = s[rows, first[has]]
            todo = todo[~has]
            if len(todo) == 0:
                break
    keep = ~torch.isnan(res).any(1)
    if not keep.all():
        warnings.warn(f"{int((~keep).sum())} of {n} draws had no sample inside the prior box; dropped")
    return res, keep


def sample_two_step(posterior_AA, posterior_TT, data_AA, data_TT, n_samples=500,
                    show_progress_bars=False, fast=True):
    """Two-step inference of the paper.

    1. sample the IMS noise amplitude ``P`` from the TT posterior using the
       2970 unbinned TT points;
    2. for each ``P`` draw one sample of the 28 AA parameters conditioned on
       the AA data with that ``P`` appended (marginalises over ``P``).

    ``fast=True`` samples the AA flow for all ``P`` at once (seconds);
    ``fast=False`` does one sbi ``posterior.sample`` call per ``P`` as in the
    original notebook (minutes, same distribution).

    Returns ``(theta_samples (n, 28), P_samples (n,))`` as NumPy arrays.
    """
    import torch

    x_TT = torch.as_tensor(np.asarray(data_TT[:2970]), dtype=torch.float32)
    P = posterior_TT.sample((n_samples,), x=x_TT, show_progress_bars=show_progress_bars).reshape(-1)
    x_AA = torch.as_tensor(np.asarray(data_AA[:3970]), dtype=torch.float32)
    if fast:
        contexts = torch.cat([x_AA.expand(len(P), -1), P.reshape(-1, 1)], dim=1)
        theta, keep = _flow_sample_in_box(posterior_AA, contexts, n_dims_box=27)
        return theta[keep].numpy(), P[keep].numpy()
    thetas = [posterior_AA.sample((1,), x=torch.cat([x_AA, p.reshape(1)]),
                                  show_progress_bars=False).reshape(-1)
              for p in P]
    return torch.stack(thetas).numpy(), P.numpy()


def to_physical(theta_samples):
    """Unit-cube samples -> dict with ``slopes`` (n, 26), ``log10_amp`` (n,) and ``A`` (n,)."""
    theta_samples = np.asarray(theta_samples)
    out = {
        "slopes": -12 + theta_samples[:, :26] * 24,
        "log10_amp": -14 + theta_samples[:, 26] * 8,
        "A": theta_samples[:, 27],
    }
    return out


def spectra_from_samples(theta_samples, f, breakpoints):
    """``h^2 Omega_GW(f)`` for every posterior sample, shape ``(n, len(f))``."""
    p = to_physical(theta_samples)
    return piecewise_powerlaw(f, p["slopes"], 10.0 ** p["log10_amp"], breakpoints)


def pointwise_bands(spectra, levels=(0.68, 0.95)):
    """Median and pointwise equal-tailed credible bands of the spectra.

    Returns ``median, {level: (low, high)}``. This is the usual way to draw
    a reconstruction band (it accounts for correlations between slopes).
    """
    median = np.median(spectra, axis=0)
    bands = {lv: tuple(np.quantile(spectra, [(1 - lv) / 2, (1 + lv) / 2], axis=0)) for lv in levels}
    return median, bands


def parameter_hdi_spectra(theta_samples, f, breakpoints, levels=(0.68268, 0.95450, 0.99730)):
    """Reproduces the original ``Plots.py`` curves.

    Takes the per-parameter HDI limits of the slopes and amplitude and draws
    the spectrum through all lower limits and all upper limits, plus the
    spectrum of the posterior-mean parameters. Returns
    ``mean_curve, {level: (low_curve, high_curve)}``.
    """
    from .plot_signal import hdi

    p = to_physical(theta_samples)
    mean_curve = piecewise_powerlaw(f, p["slopes"].mean(0), 10.0 ** p["log10_amp"].mean(), breakpoints)
    out = {}
    for lv in levels:
        lims = np.array([hdi(p["slopes"][:, i], lv) for i in range(26)] + [hdi(p["log10_amp"], lv)])
        lo = piecewise_powerlaw(f, lims[:26, 0], 10.0 ** lims[26, 0], breakpoints)
        hi = piecewise_powerlaw(f, lims[:26, 1], 10.0 ** lims[26, 1], breakpoints)
        out[lv] = (lo, hi)
    return mean_curve, out
