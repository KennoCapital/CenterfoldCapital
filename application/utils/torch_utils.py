import torch

def N_cdf(x):
    return torch.distributions.Normal(loc=0.0, scale=1.0).cdf(x)

def max0(x):
    return torch.maximum(x, torch.tensor(0.0))

def smoothing(type : str, x : torch.Tensor, barrier : torch.Tensor, lb : torch.Tensor, ub : torch.Tensor):
    if type == None:
        return torch.tensor(0.0)
    if type == 'linear':
        return (ub - x)/(ub - lb)
    if type == 'sigmoid':
        slope = (0 - max0(lb)) / (ub - lb)
        return 1/(1+torch.exp(-slope * (x - barrier)/(barrier - lb)))
    else:
        raise ValueError('Wrongly specified smoothing method')

