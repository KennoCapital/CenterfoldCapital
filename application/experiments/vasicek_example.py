import numpy as np
from scipy.stats import norm

N = lambda x: norm(loc=0.0, scale=1.0).cdf(x)


class Vasicek:
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def _calc_A(self, t):
        B = self._calc_B(t)
        return (self.b - self.sigma**2 / (2*self.a**2)) * (t - B) - \
               self.sigma ** 2 * B ** 2 / (4 * self.a)

    def _calc_B(self, t):
        return  (1 - np.exp(-self.a * t)) / self.a

    def calc_zcb(self, t, r0):
        """P(0,t)=exp{ -A(0,t) - B(0,t) * r(t)}"""
        return np.exp(-self._calc_A(t) - self._calc_B(t) * r0)

    def calc_fwd(self, t, dt, r0):
        """F(0,t,t+dt) = 1/dt * (P(0,t) / P(0,t+dt) - 1)"""
        zcb_t = self.calc_zcb(t, r0)
        zcb_tdt = self.calc_zcb(t+dt, r0)
        return 1 / dt * (zcb_t / zcb_tdt - 1)

    def calc_swap_rate(self, t, dt, r0):
        zcb = self.calc_zcb(t, r0)
        return (zcb[0] - zcb[-1]) / (dt * np.sum(zcb[1:]))

    def calc_cpl(self, t, dt, r0, K):
        fwd = self.calc_fwd(t, dt, r0)
        d1 = (np.log(fwd / K) + 0.5 * self.sigma ** 2 * t) / (self.sigma * np.sqrt(t))
        d2 = (np.log(fwd / K) - 0.5 * self.sigma ** 2 * t) / (self.sigma * np.sqrt(t))
        return dt * self.calc_zcb(t+dt, r0) * ( fwd*N(d1) - K*N(d2) )

    def calc_cap(self, t, dt, r0, K):
        return np.sum(self.calc_cpl(t, dt, r0, K))


def Bl(zcb, fwd, K, sigma, t, dt):
    d1 = (np.log(fwd / K) + 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    d2 = (np.log(fwd / K) - 0.5 * sigma ** 2 * t) / (sigma * np.sqrt(t))
    cpl = dt * zcb * (fwd * N(d1) - K * N(d2))
    return cpl





if __name__ == '__main__':
    T = 10.0
    n = int(T*4 - 1)
    dt = np.array(0.25)
    t = np.linspace(dt, T, n+1, True)
    a = np.array(0.86)
    b = np.array(0.09)
    sigma = np.array(0.0562337)
    r0 = np.array(0.08)


    mld = Vasicek(a, b, sigma)

    zcb = mld.calc_zcb(t, r0)
    fwd = 1/dt * (zcb[0:-1] / zcb[1:] - 1)
    swap_rate = (zcb[0] - zcb[-1]) / (dt * np.sum(zcb[1:]))

    cpl = Bl(zcb[1:], fwd, swap_rate, sigma, t[:-1], dt)
    cap = np.sum(cpl)


