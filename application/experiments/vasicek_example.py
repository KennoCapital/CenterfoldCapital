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


if __name__ == '__main__':
    a = np.array(2.5)
    b = np.array(0.08)
    sigma = np.array(0.2)
    r0 = np.array(0.1)
    t = np.array(1.0)

    mld = Vasicek(a, b, sigma)

    zcb = mld.calc_zcb(t, r0)
    print(zcb)