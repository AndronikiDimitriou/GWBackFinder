"""Mock LISA data generation (port of ``mock_signal.jl``, ``mock_noise.jl``
and ``different_signals.jl`` from GWBackFinder.jl).

Data model (https://arxiv.org/abs/2009.11845, https://arxiv.org/abs/2309.08430)
--------------------------------------------------------------------------------
For every frequency the Julia code draws ``n_chunks = 94`` chunks of two real
Gaussian numbers for the signal and two for the noise and averages

    D(f) = mean_chunks[ Omega_s (e1^2 + e2^2)/2 + Omega_n (e3^2 + e4^2)/2 ].

Since ``sum_chunks (e1^2 + e2^2) / (2 n)`` is exactly ``Gamma(n, 1/n)``
distributed, ``method="gamma"`` (the default) draws

    D(f) = Omega_s * G1 + Omega_n * G2,   G1, G2 ~ Gamma(94, 1/94)

which has the *same distribution* as the Julia/CUDA code but needs 2 random
numbers per frequency instead of 376, so it runs fast on a CPU.
``method="gaussian"`` reproduces the chunk-by-chunk Gaussian draws literally.

Above 3 mHz the data are coarse-grained into 1000 log bins with
inverse-noise weights; below they are kept at full resolution (2970 points),
giving 3970 data points per channel.
"""
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

from .binning import N_BINS, N_EDGES_26, N_UNBINNED, binning, frequency_grid, inverse_variance_bin
from .noise import C_LIGHT, L_ARM, noise_basis
from .signal import _offsets
from .templates import template_omega

N_CHUNKS = 94
#: nominal noise parameters and their 20% Gaussian prior widths
P_MEAN, P_STD = 15.0, 0.2 * 15.0
A_MEAN, A_STD = 3.0, 0.2 * 3.0


class Simulator:
    """Holds the frequency grid, binning and noise templates.

    Build it once and call the ``simulate_*`` methods many times.

    Parameters
    ----------
    f : array, optional
        Frequency grid; default ``3e-5 : 1e-6 : 0.5`` Hz as in the paper code.
    n_chunks : int
        Number of data chunks averaged over (94 in the paper code).
    method : {"gamma", "gaussian"}
        How chunk averaging is sampled (see module docstring).
    n_edges : int
        Number of log-spaced edges of the piecewise power law. 27 matches the
        trained ``posterior_AA.pkl`` (and N_c = 27 in the paper); the GitHub
        Julia code had 26.
    """

    def __init__(self, f=None, n_chunks=N_CHUNKS, method="gamma", L=L_ARM, c=C_LIGHT,
                 n_edges=N_EDGES_26):
        if method not in ("gamma", "gaussian"):
            raise ValueError("method must be 'gamma' or 'gaussian'")
        self.f = frequency_grid() if f is None else np.asarray(f, dtype=np.float64)
        self.n_chunks = n_chunks
        self.method = method
        (self.idx, self.idx_26, self.logbins_26, self.logbins,
         self.f_filtered) = binning(self.f, n_edges)
        self.n_edges = n_edges
        # Sb1..Sb25 and the pivot Sb13
        self.breakpoints = self.logbins_26[:25]
        self.f_pivot = self.logbins_26[12]
        # cached pieces of the piecewise power law on this grid
        self._seg = np.searchsorted(self.breakpoints, self.f, side="left")
        self._pivot_segment = int(np.searchsorted(self.breakpoints, self.f_pivot, side="left"))
        self._x = np.log(self.f / self.f_pivot)
        self._log_breaks = np.log(self.breakpoints / self.f_pivot)
        # Omega_noise(P, A) = A^2 u + P^2 v
        self._uAA, self._vAA = noise_basis(self.f, "AA", L, c)
        self._uTT, self._vTT = noise_basis(self.f[:N_UNBINNED], "TT", L, c)
        self._uTT_full = None

    # ------------------------------------------------------------------ helpers
    @property
    def n_data(self):
        """Length of one channel's data vector (2970 + 1000 = 3970)."""
        return N_UNBINNED + N_BINS

    def omega_noise_AA(self, P, A):
        return A**2 * self._uAA + P**2 * self._vAA

    def omega_noise_TT(self, P, A, full=False):
        if not full:
            return A**2 * self._uTT + P**2 * self._vTT
        if self._uTT_full is None:
            self._uTT_full, self._vTT_full = noise_basis(self.f, "TT")
        return A**2 * self._uTT_full + P**2 * self._vTT_full

    def omega_signal(self, slopes, log10_amp):
        """26-segment power law with physical slopes and log10 amplitude."""
        slopes = np.asarray(slopes, dtype=np.float64)
        c = _offsets(slopes, self._log_breaks, self._pivot_segment)
        return np.exp(slopes[self._seg] * self._x + c[self._seg] + np.log(10.0) * log10_amp)

    def _chunk_average(self, n, rng):
        """Draw ``mean_chunks (e1^2 + e2^2) / 2`` for ``n`` frequencies."""
        k = self.n_chunks
        if self.method == "gamma":
            return rng.gamma(k, 1.0 / k, size=n)
        acc = np.zeros(n)
        for _ in range(k):
            acc += (rng.standard_normal(n) ** 2 + rng.standard_normal(n) ** 2) / 2
        return acc / k

    def _coarse_grain(self, data, omega_noise):
        """Keep f < 3 mHz, inverse-noise-weight average the rest into 1000 bins."""
        w = 1.0 / omega_noise[N_UNBINNED:]
        binned = inverse_variance_bin(data[N_UNBINNED:], w, self.idx)
        f_binned = inverse_variance_bin(self.f_filtered, w, self.idx)
        return (np.concatenate([data[:N_UNBINNED], binned]),
                np.concatenate([self.f[:N_UNBINNED], f_binned]))

    @staticmethod
    def _rng(rng):
        return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    # --------------------------------------------------------------- AA channel
    def simulate_AA(self, theta, rng=None, P=None, return_freq=False):
        """Signal + noise in the AA channel (Julia ``model_train_data`` / ``model_plot``).

        Parameters
        ----------
        theta : array (28,)
            Unit-cube parameters: 26 slopes and 1 amplitude in [0, 1] (mapped to
            [-12, 12] and log10 A_s in [-14, -6]) followed by the noise
            parameter ``A``. (Julia read 27 slopes but only used the first 26.)
        rng : np.random.Generator or seed, optional
        P : float, optional
            IMS noise parameter. By default drawn from N(15, 3) as in Julia.
        return_freq : bool
            Also return ``log10`` of the (weighted) frequencies, like ``model_plot``.

        Returns
        -------
        x : array (3971,)
            ``log10`` of the 3970 data points followed by ``P``.
        log10_f : array (3970,), only if ``return_freq``.
        """
        rng = self._rng(rng)
        theta = np.asarray(theta, dtype=np.float64)
        slopes = -12 + theta[:26] * 24
        log10_amp = -14 + theta[26] * 8
        A = theta[27]
        if P is None:
            P = P_MEAN + rng.standard_normal() * P_STD
        n = self.f.size
        omega_s = self.omega_signal(slopes, log10_amp)
        omega_n = self.omega_noise_AA(P, A)
        data = omega_s * self._chunk_average(n, rng) + omega_n * self._chunk_average(n, rng)
        data_total, f_total = self._coarse_grain(data, omega_n)
        x = np.append(np.log10(data_total), P)
        return (x, np.log10(f_total)) if return_freq else x

    # --------------------------------------------------------------- TT channel
    def simulate_TT(self, P, rng=None, A=None, return_freq=False):
        """Noise-only TT channel below 3 mHz (Julia ``model_noise_train_data``).

        Returns ``log10`` of the 2970 data points (and ``log10 f`` if requested).
        ``A`` is a nuisance parameter drawn from N(3, 0.6) unless given.
        """
        rng = self._rng(rng)
        P = float(np.asarray(P).reshape(-1)[0])
        if A is None:
            A = A_MEAN + rng.standard_normal() * A_STD
        data = self.omega_noise_TT(P, A) * self._chunk_average(N_UNBINNED, rng)
        x = np.log10(data)
        return (x, np.log10(self.f[:N_UNBINNED])) if return_freq else x

    # ------------------------------------------------------------ test signals
    def different_signals(self, signal, params_signal, params_noise=(P_MEAN, A_MEAN), rng=None):
        """Template signal + noise in both channels (Julia ``different_signals``).

        Parameters
        ----------
        signal : str
            One of ``powerlaw, peak, wiggly, bpowerlaw, dpeak, three_peaks``
            (Julia struct names like ``"zPeak"`` are accepted too).
        params_signal : sequence
            Parameters in the same order as the Julia ``z*`` structs, e.g.
            ``[log10_A, gamma]`` for a power law.
        params_noise : (P, A)

        Returns
        -------
        Data_AA : (3971,)  log10 AA data followed by ``P`` (posterior input)
        Data_TT : (3970,)  log10 TT data (use ``Data_TT[:2970]`` for the TT posterior)
        log10_f_AA, log10_f_TT : (3970,)
        """
        rng = self._rng(rng)
        P, A = float(params_noise[0]), float(params_noise[1])
        n = self.f.size
        omega_s = template_omega(signal, self.f, params_signal)
        signal_part = omega_s * self._chunk_average(n, rng)    # shared by both channels, as in Julia

        omega_n_AA = self.omega_noise_AA(P, A)
        data_AA, f_AA = self._coarse_grain(signal_part + omega_n_AA * self._chunk_average(n, rng), omega_n_AA)

        omega_n_TT = self.omega_noise_TT(P, A, full=True)
        # As in Julia, the TT bins are weighted with the *AA* noise: the TT noise
        # vanishes at the transfer-function zeros (f = n c / L ~ 0.12, 0.24 Hz),
        # so 1/Omega_TT weights would blow up there. Only Data_TT[:2970] enters
        # the inference anyway.
        data_TT, f_TT = self._coarse_grain(signal_part + omega_n_TT * self._chunk_average(n, rng), omega_n_AA)

        return (np.append(np.log10(data_AA), P), np.log10(data_TT),
                np.log10(f_AA), np.log10(f_TT))

    # ----------------------------------------------------------------- batches
    def simulate_AA_batch(self, thetas, seed=None, n_jobs=1, dtype=np.float32, progress=True):
        """Simulate many AA data sets; returns ``(N, 3971)``."""
        return _batch(self, "AA", np.asarray(thetas), seed, n_jobs, dtype, progress)

    def simulate_TT_batch(self, Ps, seed=None, n_jobs=1, dtype=np.float32, progress=True):
        """Simulate many TT data sets; returns ``(N, 2970)``."""
        return _batch(self, "TT", np.asarray(Ps).reshape(len(Ps), -1), seed, n_jobs, dtype, progress)


# ---------------------------------------------------------------------- batch
_WORKER_SIM = None


def _init_worker(f, n_chunks, method, n_edges):
    global _WORKER_SIM
    _WORKER_SIM = Simulator(f=f, n_chunks=n_chunks, method=method, n_edges=n_edges)


def _run(args):
    channel, params, seed, dtype = args
    sim = _WORKER_SIM
    rng = np.random.default_rng(seed)
    fn = sim.simulate_AA if channel == "AA" else sim.simulate_TT
    return np.stack([fn(p, rng) for p in params]).astype(dtype)


def _batch(sim, channel, params, seed, n_jobs, dtype, progress, chunk=256):
    n_jobs = os.cpu_count() if n_jobs in (None, -1) else n_jobs
    pieces = [params[i:i + chunk] for i in range(0, len(params), chunk)]
    seeds = np.random.SeedSequence(seed).spawn(len(pieces))
    jobs = [(channel, p, s, dtype) for p, s in zip(pieces, seeds)]
    bar = _progress(len(params), progress)
    out = []
    if n_jobs == 1:
        global _WORKER_SIM
        _WORKER_SIM = sim
        for j in jobs:
            out.append(_run(j))
            bar.update(len(j[1]))
    else:
        with ProcessPoolExecutor(n_jobs, initializer=_init_worker,
                                 initargs=(sim.f, sim.n_chunks, sim.method, sim.n_edges)) as ex:
            for j, res in zip(jobs, ex.map(_run, jobs)):
                out.append(res)
                bar.update(len(j[1]))
    bar.close()
    return np.concatenate(out) if out else np.empty((0,), dtype)


def _progress(total, enabled):
    if enabled:
        try:
            from tqdm.auto import tqdm
            return tqdm(total=total)
        except ImportError:
            pass

    class _Null:
        def update(self, n): pass
        def close(self): pass
    return _Null()
