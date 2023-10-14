import torch
import scipy
from application.utils.torch_utils import N_cdf


def black_cpl(sigma_black, zcb, fwd, K, t, delta, notional=torch.tensor(1.0)):
    """
    Black76's formula for European caplet (call option on a Forward / Libor)
    Filipovic eq. 2.6, the current time is assumed to be 0.0.

    :param zcb:             Zero coupon bond price / discount factor
    :param fwd:             Forward (spot) price
    :param K:               Strike
    :param sigma_black:     Volatility
    :param t:               Expiry / reset date
    :param delta:           Accrual period
    :param notional:
    :return:
    """
    d1 = (torch.log(fwd / K) + 0.5 * torch.pow(sigma_black, torch.tensor(2.0)) * t) / (sigma_black * torch.sqrt(t))
    d2 = (torch.log(fwd / K) - 0.5 * torch.pow(sigma_black, torch.tensor(2.0)) * t) / (sigma_black * torch.sqrt(t))
    return notional * delta * zcb * (fwd * N_cdf(d1) - K * N_cdf(d2))


def black_cpl_delta(sigma_black, zcb, fwd, K, t, delta, notional=torch.tensor(1.0)):
    d1 = (torch.log(fwd / K) + 0.5 * torch.pow(sigma_black, torch.tensor(2.0)) * t) / (sigma_black * torch.sqrt(t))
    return notional * delta * zcb * N_cdf(d1)


def black_cpl_iv(market_price, zcb, fwd, K, t, delta):
    def obj(x):
        notional = torch.tensor(1E6)  # Using a notional helps finding a solution
        black_price = torch.sum(black_cpl(torch.tensor(x, dtype=torch.float64), zcb, fwd, K, t, delta, notional))
        return  black_price - market_price * notional

    black_iv = scipy.optimize.bisect(f=obj, a=-1.0, b=5.0, maxiter=2500)
    if black_iv < torch.tensor(0.0):
        print("Warning, got negative implied black vol")
    return black_iv


def black_EuPaySwaption(sigma, zcb, swap_rate, K, t, delta, N=torch.tensor([1.0])):
    """
    Black's formula for European Payer Swaption
    Filipovic eq. 2.9, the current time is assumed to be 0.0.

    The underlying swap has timeline t = T0, ... Tn, where
        T0 is the first fixing date (and exercise date of the swaption),
        Tn-1 is the last fixing date,
        Tn is the last payment date

    :param zcb:         Zero coupon bond prices P(0, Ti), i = 1,...,n ,
                            where Ti is the cashflow dates in the underlying swap.
    :param swap_rate:   Swap_rate
    :param K:           Strike rate / Fixed rate of the underlying swap
    :param sigma:       Volatility
    :param t:           Exercise date of the swaption (T0)
    :param delta:       Accrual period (assumed constant)
    :param N:           Notional
    :return:
    """
    d1 = (torch.log(swap_rate/K) + 0.5 * sigma ** 2 * t) / (sigma * torch.sqrt(t))
    d2 = (torch.log(swap_rate/K) - 0.5 * sigma ** 2 * t) / (sigma * torch.sqrt(t))

    return N * delta * (swap_rate * N_cdf(d1) - K * N_cdf(d2)) * torch.sum(zcb)


def black_EuPaySwaption_iv(market_price, zcb, swap_rate, K, t, delta, N=torch.tensor([1.0])):
    def obj(x):
        return torch.sum(black_EuPaySwaption(x, zcb, swap_rate, K, t, delta, N)) - market_price

    return scipy.optimize.bisect(f=obj, a=1E-6, b=5.0, maxiter=1000)
