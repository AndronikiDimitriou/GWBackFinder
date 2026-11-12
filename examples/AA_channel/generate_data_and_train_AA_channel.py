#%%
"""Simulate AA-channel training data and train the signal posterior.

Replaces the old Python -> Julia (data_generation_AA_channel.jl) -> .jld2 round trip:
the data are now simulated directly in Python with GWBackFinder.Simulator.
"""
import os

import numpy as np
import torch

from GWBackFinder import Simulator
from GWBackFinder import prior as GW_prior
from GWBackFinder import train as GW_train

basepath = os.path.dirname(os.path.abspath(__file__))
n_simulations = 100      # ~1e5 in the paper
n_jobs = os.cpu_count()  # parallel simulation workers

#%%
## sample parameters from the prior: 26 slopes + amplitude in [0, 1], noise A ~ N(3, 0.6)
custom_prior = GW_prior.get_prior()
thetas = custom_prior.sample((n_simulations,))

#%%
## simulate: x = [log10 data (3970 points), P]  -> shape (n_simulations, 3971)
sim = Simulator()
gw_total = sim.simulate_AA_batch(thetas.numpy(), seed=0, n_jobs=n_jobs)
np.savez_compressed(os.path.join(basepath, "data_AA.npz"), thetas=thetas.numpy(), x=gw_total)

gw_total = torch.tensor(gw_total, dtype=torch.float32)
print(thetas.shape, gw_total.shape)

#%%
## train and save the inference object
GW_train.train(thetas=thetas, gw_total=gw_total, prior=custom_prior, resume_training=False,
               validation_fraction=0.2, learning_rate=1e-4, show_train_summary=True,
               max_num_epochs=200, path_saved=None, path_inference=basepath,
               name_file="train_200.pkl", model_type="nsf", hidden_features=64, num_transforms=3)

#%%
## build the posterior from train_200.pkl and save it
GW_train.get_posterior(os.path.join(basepath, "train_200.pkl"), os.path.join(basepath, "posterior.pkl"))
