#%%
import os

import torch
from sbi import utils as utils
from torch.distributions.normal import Normal

from GWBackFinder import train as GW_train

basepath = os.path.dirname(os.path.abspath(__file__))

#%%
prior_c = Normal(torch.tensor([15.0]), torch.tensor([0.2 * 15]))
prior, *_ = utils.process_prior(prior_c)

## resume training from train_200_noise.pkl
GW_train.train(thetas=None, gw_total=None, prior=prior, resume_training=True,
               validation_fraction=0.2, learning_rate=1e-4, show_train_summary=True,
               max_num_epochs=400, path_saved=os.path.join(basepath, "train_200_noise.pkl"),
               path_inference=basepath, name_file="train_400_noise.pkl",
               model_type="nsf", hidden_features=64, num_transforms=3)

#%%
GW_train.get_posterior(os.path.join(basepath, "train_400_noise.pkl"),
                       os.path.join(basepath, "posterior_noise.pkl"))
