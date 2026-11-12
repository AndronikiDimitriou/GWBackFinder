#%%
"""Two-step reconstruction with the paper posteriors (posterior_TT.pkl, posterior_AA.pkl).

Pure-Python version of the old juliacall notebook. Uses the paper data model
(GWBackFinder.Simulator), which is what these two posteriors were trained on.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from GWBackFinder import Simulator
from GWBackFinder.noise import Omega_noiseh2_AA, Omega_noiseh2_TT
from GWBackFinder.posterior import (load_posterior, parameter_hdi_spectra, pointwise_bands,
                                    sample_two_step, spectra_from_samples)
from GWBackFinder.templates import template_omega

basepath = os.path.dirname(os.path.abspath(__file__))
sim = Simulator()

#%%
## test signals (same parameter order as the Julia z* structs)
test_signals = {
    "powerlaw":    [-10.0, 0.6],
    "peak":        [-9.0, 0.002, 0.2],
    "wiggly":      [-10.0, 2.0, 0.1],
    "bpowerlaw":   [-11.0, -1.0, 2 / 3, 0.002],
    "dpeak":       [-11.0, -10.0, 0.001, 0.01, 0.1, 0.1],
    "three_peaks": [-10.0, -10.0, -10.0, 5e-4, 2e-3, 8e-3, 0.1, 0.1, 0.1],
}
signal = "powerlaw"
params_noise = [15.0, 3.0]          # P, A

Data_AA, Data_TT, f_total_AA, f_total_TT = sim.different_signals(
    signal, test_signals[signal], params_noise, rng=0)

#%%
plt.figure()
plt.plot(f_total_AA, Data_AA[:3970], label="AA")
plt.plot(f_total_TT, Data_TT[:3970], label="TT")
plt.xlabel("log10 f [Hz]")
plt.ylabel(r"log10 $h^2\Omega$")
plt.legend()
plt.show()

#%%
posterior = load_posterior(os.path.join(basepath, "AA_channel", "posterior_AA.pkl"))
posterior_noise = load_posterior(os.path.join(basepath, "TT_channel", "posterior_TT.pkl"))

samples, samples_TT = sample_two_step(posterior, posterior_noise, Data_AA, Data_TT, n_samples=500)
plt.hist(samples_TT, bins=50)
plt.axvline(params_noise[0], color="k")
plt.xlabel("P")
plt.show()

#%%
f = sim.f[::50]
spectra = spectra_from_samples(samples, f, sim.breakpoints)
median, bands = pointwise_bands(spectra)
mean_curve, hdi_curves = parameter_hdi_spectra(samples, f, sim.breakpoints)   # curves as in the old notebook

pls = np.genfromtxt(os.path.join(basepath, "plis_LISA_new_noise.dat"))
fig, ax = plt.subplots()
ax.loglog(10 ** pls[15:, 0], 10 ** pls[15:, 1], "k--", label="PLS")
ax.loglog(f, Omega_noiseh2_TT(f, 2.5e9, 3e8, *params_noise), ls="--", label="TT noise")
ax.loglog(f, Omega_noiseh2_AA(f, 2.5e9, 3e8, *params_noise), "g--", label="AA noise")
ax.loglog(f, template_omega(signal, f, test_signals[signal]), "r--", label="injected signal")
ax.loglog(f, median, "b", label="reconstructed (median)")
ax.fill_between(f, *bands[0.68], alpha=0.7, color="#838B8B", label="68% CI")
ax.fill_between(f, *bands[0.95], alpha=0.5, color="#808080", label="95% CI")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"$h^2\Omega$")
ax.set_ylim(1e-14, 1e-4)
ax.legend(loc="lower right", fontsize=8)
plt.show()
