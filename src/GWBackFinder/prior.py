import torch
from sbi.utils import process_prior

## Define Prior 
## A prior class that mimicks the behaviour of a torch.distributions.Distribution class.
## Then sbi will wrap this class to make it a fully functional torch Distribution.
## See https://sbi-dev.github.io/sbi/faq/question_07/

class CustomUniformPrior:
    """User defined numpy uniform prior.

    Custom prior with user-defined valid .sample and .log_prob methods.
    """

    def __init__(self, lower, upper,lower1,lower2, return_numpy: bool = False):
        """
        Initialize the CustomUniformPrior. CustomUniformPrior is uniform prior from 0 to 1 for the slopes and the amplitude of the GW signal
        and gaussian for the noise parameter A.

        Parameters:
        - lower: Lower bounds for the uniform distributions.
        - upper: Upper bounds for the uniform distributions.
        - lower1: Parameter for mean of normal distribution .
        - lower2: Parameter for standard deviation of normal distribution.
        - return_numpy: If True, return samples and log probabilities as NumPy arrays.
        """ 
        self.lower = lower
        self.upper = upper
        self.dist = torch.distributions.uniform.Uniform(lower, upper)
        self.normal= torch.distributions.normal.Normal(3*lower1, 3*.2*lower2)

        self.return_numpy = return_numpy

    def sample(self, nsamples=1):
        nsamples= torch.atleast_1d(torch.tensor(nsamples))
        samples = torch.hstack([self.dist.sample(nsamples).reshape(*nsamples,-1),self.normal.sample(nsamples).reshape(*nsamples,-1)])
        return samples.numpy() if self.return_numpy else samples

    def log_prob(self, values):
        if self.return_numpy:
            values = torch.as_tensor(values)
        log_probs = torch.sum(self.dist.log_prob(values[...,:27]),dim=-1) + self.normal.log_prob(values[...,27])
        return log_probs.numpy() if self.return_numpy else log_probs

    def bounds(self):
        lower_bound_vals=torch.hstack((torch.zeros(27),torch.tensor([-10000])))
        upper_bound_vals=torch.hstack((torch.ones(27),torch.tensor([10000])))
        return lower_bound_vals,upper_bound_vals

## In sbi it is sometimes required to check the support of the prior, e.g.,
## when the prior support is bounded and one wants to reject samples from the posterior density estimator that lie outside the prior support.
## In torch Distributions this is handled automatically, however, when using a custom prior it is not. 
## See https://sbi-dev.github.io/sbi/faq/question_07/

def get_prior():
    """
        Return prior with correct support.
    """ 
    prior =  CustomUniformPrior(torch.zeros(27), torch.ones(27),torch.ones(1),torch.ones(1))
    lower_bound_vals,upper_bound_vals=prior.bounds()
    custom_prior, *_  = process_prior(prior,
                      custom_prior_wrapper_kwargs=dict(lower_bound=lower_bound_vals,
                                                       upper_bound=upper_bound_vals))
    return custom_prior


