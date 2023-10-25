import torch


def annuity():
    raise NotImplementedError


def zcb_yield_to_price(t, y):
    return torch.exp(-y * t)

def zcb_price_to_yield(t, zcb):
    return - torch.log(zcb) / t


def forward(zcb1, zcb2, delta):
    """F(t,T,T+delta) = 1 / delta * ( P(t,T) / P(t,T+delta) - 1 )"""
    return 1 / delta * (zcb1 / zcb2 - 1)


def swap(zcb:       torch.Tensor,
         delta:     torch.Tensor,
         K:         torch.Tensor or None = None,
         N:         torch.Tensor = torch.tensor(1.0)):
    """S(0) = N * [ P(0,T0) - P(0,Tn) - K * delta * sum_{i=1}^n P(0,Ti) ]"""
    if K is None:
        K = swap_rate(zcb, delta)
    if zcb.dim() == 2:
        return N * (zcb[0, :] - zcb[-1, :] - K * delta * torch.sum(zcb[1:, :], dim=0))
    return N * (zcb[0] - zcb[-1] - K * delta * torch.sum(zcb[1:], dim=0))


def swap_rate(zcb: torch.Tensor, delta: torch.Tensor):
    """R(0) = [ P(0,T0) - P(0,Tn) ] / [delta * sum_{i=1}^n P(0,Ti) ]"""
    if zcb.dim() == 2:
        return (zcb[0, :] - zcb[-1, :]) / (delta * torch.sum(zcb[1:], dim=0))
    return (zcb[0] - zcb[-1]) / (delta * torch.sum(zcb[1:]))


def forward_rate_agreement(zcb: torch.Tensor,
                           fwd: torch.Tensor,
                           delta: torch.Tensor,
                           K: torch.Tensor,
                           N: torch.Tensor = torch.tensor(1.0)):
    return N * delta * (fwd - K) * zcb
