#%%
import os

from GWBackFinder import prior as GW_prior
from GWBackFinder import train as GW_train

basepath = os.path.dirname(os.path.abspath(__file__))

#%%
custom_prior = GW_prior.get_prior()

## resume training from train_200.pkl
GW_train.train(thetas=None, gw_total=None, prior=custom_prior, resume_training=True,
               validation_fraction=0.2, learning_rate=1e-4, show_train_summary=True,
               max_num_epochs=400, path_saved=os.path.join(basepath, "train_200.pkl"),
               path_inference=basepath, name_file="train_400.pkl",
               model_type="nsf", hidden_features=64, num_transforms=3)

#%%
GW_train.get_posterior(os.path.join(basepath, "train_400.pkl"), os.path.join(basepath, "posterior.pkl"))
