#%%
"""Make the two summary figures in docs/:

* docs/reconstructions.png - four test signals reconstructed with the trained
  posteriors (two-step TT -> P, AA | P)
* docs/calibration.png     - simulation-based calibration (SBC) of the AA and TT networks

Run from the repository root:  python examples/make_figures.py
Takes a few minutes on a laptop (mostly the calibration part).
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from GWBackFinder import Simulator
from GWBackFinder.noise import Omega_noiseh2_AA
from GWBackFinder.posterior import (_flow_sample_in_box, load_posterior, pointwise_bands,
                                    sample_two_step, spectra_from_samples)
from GWBackFinder.prior import get_prior
from GWBackFinder.templates import template_omega

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "..", "docs")
os.makedirs(out_dir, exist_ok=True)

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3de"

posterior_AA = load_posterior(os.path.join(here, "AA_channel", "posterior_AA.pkl"))
posterior_TT = load_posterior(os.path.join(here, "TT_channel", "posterior_TT.pkl"))
sim = Simulator()
torch.manual_seed(0)

#%% ------------------------------------------------------- reconstructions
tests = {
    "powerlaw": [-10.0, 0.6],
    "peak": [-9.0, 0.002, 0.2],
    "wiggly": [-10.0, 2.0, 0.1],
    "bpowerlaw": [-11.0, -1.0, 2 / 3, 0.002],
}
pls = np.genfromtxt(os.path.join(here, "plis_LISA_new_noise.dat"))
f = np.logspace(np.log10(3e-5), np.log10(0.5), 400)

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
for ax, (name, par) in zip(axes.ravel(), tests.items()):
    d_AA, d_TT, logf_AA, _ = sim.different_signals(name, par, (15.0, 3.0), rng=1)
    theta, P = sample_two_step(posterior_AA, posterior_TT, d_AA, d_TT, n_samples=1000)
    median, bands = pointwise_bands(spectra_from_samples(theta, f, sim.breakpoints))
    ax.loglog(10**logf_AA, 10 ** d_AA[:3970], color="0.8", lw=0.5, label="AA data")
    ax.loglog(10 ** pls[15:, 0], 10 ** pls[15:, 1], "k:", lw=1, label="PLS")
    ax.loglog(f, Omega_noiseh2_AA(f, 2.5e9, 3e8, 15, 3), "g--", lw=1, label="AA noise")
    ax.fill_between(f, *bands[0.95], color="#9bb7d4", label="95% CI")
    ax.fill_between(f, *bands[0.68], color="#4a7fb5", label="68% CI")
    ax.loglog(f, median, "b", lw=1.5, label="reconstructed (median)")
    ax.loglog(f, template_omega(name, f, par), "r--", lw=1.5, label="injected")
    ax.set_title(f"{name}   (P = {P.mean():.2f} ± {P.std():.2f}, true 15)", fontsize=10)
    ax.set_ylim(1e-14, 1e-4)
for ax in axes[1]:
    ax.set_xlabel("Frequency [Hz]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$h^2\Omega_{GW}$")
axes[0, 0].legend(fontsize=7, loc="upper left")
fig.suptitle("Reconstruction with posterior_TT.pkl / posterior_AA.pkl")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "reconstructions.png"), dpi=130)

#%% --------------------------------------------------------- calibration
N, S = 300, 500
thetas = get_prior().sample((N,))
X = torch.tensor(sim.simulate_AA_batch(thetas.numpy(), seed=0, n_jobs=os.cpu_count()))
R = np.zeros((N, 28))
for i in range(N):
    s, keep = _flow_sample_in_box(posterior_AA, X[i].expand(S, -1), n_dims_box=27)
    R[i] = (s[keep].numpy() < thetas[i].numpy()).mean(0)
P_true = np.random.default_rng(1).normal(15, 3, (N, 1))
XT = torch.tensor(sim.simulate_TT_batch(P_true, seed=1, n_jobs=os.cpu_count()))
RT = np.array([(posterior_TT.sample((S,), x=XT[i], show_progress_bars=False).numpy().ravel() < P_true[i, 0]).mean()
               for i in range(N)])

names = [str(i) for i in range(26)] + ["amp", "A"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(12, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)

ax = fig.add_subplot(gs[0, :])
se = np.sqrt(1 / 12 / N)
ax.axhspan(0.5 - 2 * se, 0.5 + 2 * se, color=GRID, lw=0, label=f"±2σ expected if calibrated ({N} sims)")
ax.axhline(0.5, color=MUTED, lw=1)
ax.errorbar(np.arange(28), R.mean(0), yerr=2 * se, fmt="o", ms=4.5, color=BLUE, capsize=0, label="AA network")
ax.set_xticks(np.arange(28))
ax.set_xticklabels(names)
ax.set_xlabel("parameter (slopes 0–25, amplitude, noise A)")
ax.set_ylabel("mean SBC rank")
ax.set_title("AA network: mean rank of the true value among posterior samples (ideal 0.5)", loc="left", fontsize=10)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

ax = fig.add_subplot(gs[1, 0])
lv = np.linspace(0.01, 0.99, 50)
band = 1.96 * np.sqrt(lv * (1 - lv) / N)
ax.fill_between(lv, lv - band, lv + band, color=GRID, lw=0, label="95% band, calibrated")
cov = lambda r: [((r > (1 - l) / 2) & (r < (1 + l) / 2)).mean() for l in lv]
for j in range(26):
    ax.plot(lv, cov(R[:, j]), color="#b9b8b1", lw=0.8, label="slopes" if j == 0 else None)
ax.plot(lv, cov(R[:, 26]), color=BLUE, lw=2, label="amplitude")
ax.plot(lv, cov(R[:, 27]), color=ORANGE, lw=2, label="noise A")
ax.plot(lv, cov(RT), color=INK, lw=1.4, ls="--", label="TT network (P)")
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("credible level")
ax.set_ylabel("fraction of truths inside")
ax.set_title(f"Coverage ({N} sims)", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")

for k, (j, c, t) in enumerate([(26, BLUE, "amplitude"), (27, ORANGE, "noise A")]):
    ax = fig.add_subplot(gs[1, 1 + k])
    nb = 15
    e = N / nb
    s = np.sqrt(e * (1 - 1 / nb))
    ax.axhspan(e - 2 * s, e + 2 * s, color=GRID, lw=0)
    ax.axhline(e, color=MUTED, lw=0.8)
    h, b = np.histogram(R[:, j], bins=nb, range=(0, 1))
    ax.bar(b[:-1], h, width=1 / nb, align="edge", color=c, edgecolor="white", linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("rank of true value")
    ax.set_title(f"Rank histogram: {t}", loc="left", fontsize=10)

fig.suptitle("Calibration of the trained posteriors (grey = perfectly calibrated)", x=0.06, ha="left", fontsize=12)
fig.savefig(os.path.join(out_dir, "calibration.png"), dpi=140, bbox_inches="tight")
print("saved figures to", os.path.abspath(out_dir))
