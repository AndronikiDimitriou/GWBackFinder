#%%
"""Simulate TT-channel (noise only, f < 3 mHz) data and train the posterior for P.

Replaces the old Python -> Julia (data_generation_TT_channel.jl) -> .jld2 round trip.
"""
import os

import numpy as np
import torch
from sbi import utils as utils
from torch.distributions.normal import Normal

from GWBackFinder import Simulator
from GWBackFinder import train as noise_train

basepath = os.path.dirname(os.path.abspath(__file__))
n_simulations = 100
n_jobs = os.cpu_count()

#%%
## prior on the IMS noise amplitude P ~ N(15, 3)
prior_c = Normal(torch.tensor([15.0]), torch.tensor([0.2 * 15]))
prior, *_ = utils.process_prior(prior_c)
thetas = prior.sample((n_simulations,))

#%%
## simulate: log10 TT data for the 2970 unbinned frequencies -> (n_simulations, 2970)
sim = Simulator()
gw_total = sim.simulate_TT_batch(thetas.numpy(), seed=0, n_jobs=n_jobs)
np.savez_compressed(os.path.join(basepath, "data_TT.npz"), thetas=thetas.numpy(), x=gw_total)

gw_total = torch.tensor(gw_total, dtype=torch.float32)
print(thetas.shape, gw_total.shape)

#%%
noise_train.train(thetas=thetas, gw_total=gw_total, prior=prior, resume_training=False,
                  validation_fraction=0.2, learning_rate=1e-4, show_train_summary=True,
                  max_num_epochs=200, path_saved=None, path_inference=basepath,
                  name_file="train_200_noise.pkl", model_type="nsf", hidden_features=64, num_transforms=3)

#%%
noise_train.get_posterior(os.path.join(basepath, "train_200_noise.pkl"),
                          os.path.join(basepath, "posterior_noise.pkl"))
