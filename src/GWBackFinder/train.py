"""Training helpers around sbi's SNPE (unchanged logic from the original repo)."""
import os
import pickle

from sbi.inference import SNPE

try:  # sbi <= 0.22
    from sbi.utils.get_nn_models import posterior_nn
except ImportError:  # newer sbi
    from sbi.neural_nets import posterior_nn


def train(thetas=None, gw_total=None, prior=None, resume_training=False, validation_fraction=0.2, 
          learning_rate=1e-4, show_train_summary=True, max_num_epochs=None, path_saved=None, path_inference=None,
          model_type="nsf", hidden_features=64, num_transforms=3, name_file=None,
          embedding_net=None, training_batch_size=50):
    
    """
    Train a density estimator with Sequential Neural Posterior Estimation (SNPE).

    Parameters:
    
    - thetas: Parameters for the simulations.
    - gw_total: Observed data.
    - prior: Prior distribution.
    - resume_training: If True, resume training from a saved model.
    - validation_fraction: Fraction of the data used for validation during training.
    - learning_rate: Learning rate for the optimizer during training.
    - show_train_summary: If True, show training summary.
    - max_num_epochs: Maximum number of epochs for training.
    - path_saved: Path to the saved model if resume_training is True.
    - path_inference: Path to save inference.
    - model_type: Type of neural network model.
    - hidden_features: Number of hidden features in the neural network model.
    - num_transforms: Number of transformations in the neural network model.
    - name_file: Name of the pickle file to save the inference object.
    - embedding_net: optional torch module compressing x before the flow
      (see default_embedding).
    - training_batch_size: mini-batch size.
    Returns:
    Trained density estimator.
    """

    if resume_training==True:
        from .posterior import load_posterior

        inference = load_posterior(path_saved)

        density_estimator = inference.train(resume_training=True, validation_fraction=validation_fraction, 
                                            learning_rate=learning_rate, show_train_summary=show_train_summary, 
                                            max_num_epochs=max_num_epochs, force_first_round_loss=True,
                                            training_batch_size=training_batch_size)

        with open(os.path.join(path_inference, name_file), "wb") as handle:
            pickle.dump(inference, handle) 
    
    else: 
        density_estimator_build_fun = posterior_nn(model=model_type, hidden_features=hidden_features, 
                                                    num_transforms=num_transforms,
                                                   **({} if embedding_net is None else {"embedding_net": embedding_net}))
        inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)

        inference = inference.append_simulations(thetas, gw_total)

        density_estimator = inference.train(resume_training=False, validation_fraction=validation_fraction, 
                                            learning_rate=learning_rate, show_train_summary=show_train_summary, 
                                            max_num_epochs=max_num_epochs, force_first_round_loss=False,
                                            training_batch_size=training_batch_size)
        
        with open(os.path.join(path_inference, name_file), "wb") as handle:
            pickle.dump(inference, handle) 
    
    return density_estimator    

def default_embedding(input_dim, output_dim=128, num_layers=3, num_hiddens=512):
    """Fully connected embedding net for long data vectors."""
    try:
        from sbi.neural_nets.embedding_nets import FCEmbedding
    except ImportError:
        from sbi.neural_nets import FCEmbedding
    return FCEmbedding(input_dim=input_dim, output_dim=output_dim,
                       num_layers=num_layers, num_hiddens=num_hiddens)


def get_posterior(path,name_file):
    """
    Load a trained inference object from a pickle file, build the posterior,
    and save the posterior to another pickle file.

    Parameters:
    - path: Path to the saved inference object pickle file.
    - name_file: Path of the pickle file to save the posterior to.

    Returns:
    - posterior: Built posterior object.
    """
    from .posterior import load_posterior

    inference = load_posterior(path)
    posterior = inference.build_posterior()

    with open(name_file, "wb") as handle:
        pickle.dump(posterior, handle)
    return posterior
