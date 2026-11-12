#%%
"""Simulation-based calibration (SBC) of the trained AA and TT posteriors.

For N simulations from the prior, record where the true parameter falls among
posterior samples (its rank). A calibrated network gives uniform ranks (mean 0.5).
To test with Julia-generated data instead, load your own (thetas, x) arrays
into `thetas_AA, X_AA` below.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from GWBackFinder import Simulator
from GWBackFinder.prior import get_prior
from GWBackFinder.posterior import load_posterior, _flow_sample_in_box

basepath = os.path.dirname(os.path.abspath(__file__))
N, S = 300, 500          # simulations, posterior samples per simulation

posterior_AA = load_posterior(os.path.join(basepath, "AA_channel", "posterior_AA.pkl"))
posterior_TT = load_posterior(os.path.join(basepath, "TT_channel", "posterior_TT.pkl"))
sim = Simulator()

#%%
## AA: data with the true P appended, as in training
thetas_AA = get_prior().sample((N,))
X_AA = torch.tensor(sim.simulate_AA_batch(thetas_AA.numpy(), seed=0, n_jobs=os.cpu_count()))
ranks = np.zeros((N, 28))
for i in range(N):
    s, keep = _flow_sample_in_box(posterior_AA, X_AA[i].expand(S, -1), n_dims_box=27)
    ranks[i] = (s[keep].numpy() < thetas_AA[i].numpy()).mean(0)

#%%
## TT
P = np.random.default_rng(1).normal(15, 3, (N, 1))
X_TT = torch.tensor(sim.simulate_TT_batch(P, seed=1, n_jobs=os.cpu_count()))
ranks_TT = np.array([(posterior_TT.sample((S,), x=X_TT[i], show_progress_bars=False).numpy().ravel() < P[i, 0]).mean()
                     for i in range(N)])

#%%
se = np.sqrt(1 / 12 / N)
names = [str(i) for i in range(26)] + ["amp", "A"]
pvals = [stats.kstest(ranks[:, j], "uniform").pvalue for j in range(28)]
print("AA parameters failing KS (p<0.01):", [names[j] for j in range(28) if pvals[j] < 0.01])
print("TT KS p-value:", stats.kstest(ranks_TT, "uniform").pvalue)

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axhspan(0.5 - 2 * se, 0.5 + 2 * se, color="0.9", label="±2σ if calibrated")
ax.axhline(0.5, color="0.5", lw=1)
ax.errorbar(np.arange(28), ranks.mean(0), yerr=2 * se, fmt="o", ms=4, color="#2a78d6", capsize=0)
ax.set_xticks(np.arange(28))
ax.set_xticklabels(names)
ax.set_ylabel("mean SBC rank")
ax.set_title("AA network calibration (ideal 0.5)", loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
