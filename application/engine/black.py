import torch
import scipy
from application.utils.torch_utils import N_cdf


def black_cpl(sigma, zcb, fwd, K, t, delta):
    """
    Black76's formula for European caplet (call option on a Forward / Libor)
    Filipovic eq. 2.6, the current time is assumed to be 0.0.

    :param zcb:     Zero coupon bond price / discount factor
    :param fwd:     Forward (spot) price
    :param K:       Strike
    :param sigma:   Volatility
    :param t:       Expiry / reset date
    :param delta:   Accrual period
    :return:
    """
    d1 = (torch.log(fwd / K) + 0.5 * sigma ** 2 * t) / (sigma * torch.sqrt(t))
    d2 = (torch.log(fwd / K) - 0.5 * sigma ** 2 * t) / (sigma * torch.sqrt(t))
    return delta * zcb * (fwd * N_cdf(d1) - K * N_cdf(d2))


def black_iv(market_price, zcb, fwd, K, t, delta):
    def obj(x):
        return torch.sum(black_cpl(x, zcb, fwd, K, t, delta)) - market_price

    return scipy.optimize.bisect(f=obj, a=1E-6, b=5.0, maxiter=1000)

