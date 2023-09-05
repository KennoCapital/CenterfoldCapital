import torch
import scipy

N_cdf = lambda x: torch.distributions.Normal(loc=0.0, scale=1.0).cdf(x)


class Vasicek:
    """
        dr(t) = a*[b-r(t)]*dt + sigma*dW(t)
    """
    def __init__(self, a, b, sigma, use_ATS=False):
        """
        :param a:           Mean reversion rate
        :param b:           Long term mean rate
        :param sigma:       Volatility,
        :param use_ATS:     Use Affine Term Structure specification to calculate ZCB prices
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.use_ATS = use_ATS

    def _calc_fwd_vol(self, t):
        """sigma(0,t) = sigma * exp{ -a * t }"""
        return self.sigma * torch.exp(-self.a * t)

    def _calc_A(self, t):
        B = self._calc_B(t)
        return (self.b - self.sigma**2 / (2*self.a**2)) * (t - B) - self.sigma ** 2 * B ** 2 / (4 * self.a)

    def _calc_B(self, t):
        return (1 - torch.exp(-self.a * t)) / self.a

    def calc_zcb(self, r0, t):
        """P(0, t)=exp{ -A(0,t) - B(0,t) * r(t)}"""
        if self.use_ATS:
            return torch.exp(-self._calc_A(t) - self._calc_B(t) * r0)

        sigma_fwd = self._calc_fwd_vol(t)
        return torch.exp(
            (r0 - self.b) * (torch.exp(-self.a * t) - 1) / self.a - \
            self.b * t + sigma_fwd**2 * t / (2 * self.a ** 2) + \
            sigma_fwd ** 2 * (4 * torch.exp(-self.a * t) - torch.exp(-2 * self.a * t) - 3) / (4 * self.a**3)
        )

    def calc_fwd(self, r0, t, delta):
        """F(0; t, t+delta) = 1/delta * (P(0,t) / P(0,t+delta) - 1)"""
        zcb_t = self.calc_zcb(r0, t)
        zcb_tdt = self.calc_zcb(r0, t+delta)
        return 1 / delta * (zcb_t / zcb_tdt - 1)

    def calc_swap_rate(self, r0, t, delta):
        """R(0) = [ P(0,T0) - P(0,Tn) ] / [delta * sum_{i=1}^n P(0,Ti) ]"""
        zcb = self.calc_zcb(r0, t)
        return (zcb[0] - zcb[-1]) / (delta * torch.sum(zcb[1:]))

    def calc_cpl(self, r0, t, delta, K):
        """
           Solution to Filipovic's prop. 7.2
                Cpl(0; t, t+delta) = P(0,t) * N(d1) - P(0,t+delta) / K_bar * N(d2)
        """
        zcb = self.calc_zcb(r0, t)

        K_bar = 1 / (1+delta*K)
        vol_integral = self.sigma**2 / (2*self.a**3) * (
                1 - torch.exp(-2*self.a*t) + torch.exp(-2*self.a*delta) - torch.exp(-2*self.a*(t+delta)) - \
                2 * (torch.exp(-self.a*delta) - torch.exp(-self.a*(2*t+delta)))
        )[:-1]
        d1 = (torch.log(zcb[:-1] * K_bar / zcb[1:]) + 0.5 * vol_integral) / torch.sqrt(vol_integral)
        d2 = d1 - torch.sqrt(vol_integral)

        return zcb[:-1] * N_cdf(d1) - zcb[1:] / K_bar * N_cdf(d2)

    def calc_cap(self, r0, t, delta, K):
        """Cp(0, t, t+delta) = sum_{i=1}^n Cpl(t; Ti_1, Ti) """
        return torch.sum(self.calc_cpl(r0, t, delta, K))


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
    cpl = delta * zcb * (fwd * N_cdf(d1) - K * N_cdf(d2))
    return cpl

def black_iv(market_price, zcb, fwd, K, t, delta):
    def obj(x):
        return torch.sum(black_cpl(x, zcb, fwd, K, t, delta)) - market_price

    sigma_iv = scipy.optimize.bisect(f=obj, a=1E-6, b=5.0, maxiter=1000)
    return sigma_iv


if __name__ == '__main__':
    torch.set_printoptions(precision=12)
    T = 30.0
    delta = 0.25
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    n = int(T / delta - 1)
    t = torch.linspace(start=delta, end=T, steps=n+1)

    # Calculate ATM cap price
    mld = Vasicek(a, b, sigma, use_ATS=True)
    swap_rate = mld.calc_swap_rate(r0, t, delta)
    cap = mld.calc_cap(r0, t, delta, swap_rate)
    print(cap)

    # Calculate ATM Implied Volatility
    zcb = mld.calc_zcb(r0, t)
    fwd = (zcb[:-1] / zcb[1:] - 1) / delta

    sigma_iv = black_iv(cap, zcb=zcb[:-1], fwd=fwd, K=swap_rate, t=t[:-1], delta=delta)
    print(sigma_iv)
