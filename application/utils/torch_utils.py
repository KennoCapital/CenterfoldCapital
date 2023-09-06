import torch


def N_cdf(x):
    return torch.distributions.Normal(loc=0.0, scale=1.0).cdf(x)
