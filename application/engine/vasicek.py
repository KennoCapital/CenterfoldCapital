import torch
import scipy
from application.utils.torch_utils import N_cdf
from application.engine.mcBase import Model, SampleDef


class Vasicek(Model):
    """
        dr(t) = a*[b-r(t)]*dt + sigma*dW(t)
    """
    def __init__(self, a, b, sigma, r0=None, use_ATS=False):
        """
        :param a:           Mean reversion rate
        :param b:           Long term mean rate
        :param sigma:       Volatility,
        :param use_ATS:     Use Affine Term Structure specification to calculate ZCB prices
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0
        self.use_ATS = use_ATS

        # Attributes for Monte Carlo
        self._timeline = None
        self._defline = None
        self._disc_curve = None
        self._x = None
        self._zcb = None
        self._fwd = None


    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def disc_curve(self):
        return self._disc_curve

    @property
    def x(self):
        return self._x

    @property
    def zcb(self):
        return self._zcb

    @property
    def fwd(self):
        return self._fwd

    def allocate(self, prdTimeline: torch.Tensor, prdDefline: SampleDef, N):
        self._timeline = prdTimeline

        if 0.0 not in prdTimeline:  # Today on timeline
            self._timeline = torch.concat([torch.tensor([0.0]), self._timeline], dim=0)
        self._defline = prdDefline

        # Calculate discount curve
        self._disc_curve = self.calc_zcb(self.r0, prdDefline.zcbMats).reshape(-1, 1)

        # Allocate state variables (short rate), zcb and fwd
        self._x = torch.full(size=(len(self.timeline), N), fill_value=torch.nan)
        self._zcb = torch.full(size=(len(self.defline.zcbMats), N), fill_value=torch.nan)
        self._fwd = torch.full(size=(len(self.defline.fwdMats), N), fill_value=torch.nan)

    def _calc_fwd_vol(self, t):
        """sigma(0,t) = sigma * exp{ -a * t }"""
        return self.sigma * torch.exp(-self.a * t)

    def _calc_A(self, t):
        B = self._calc_B(t)
        return (self.b - self.sigma ** 2 / (2 * self.a ** 2)) * (t - B) - self.sigma ** 2 * B ** 2 / (4 * self.a)

    def _calc_B(self, t):
        return (1 - torch.exp(-self.a * t)) / self.a

    def calc_zcb(self, r0, t):
        """P(0, t)=exp{ -A(0,t) - B(0,t) * r(t)}"""
        if self.use_ATS:
            return torch.exp(-self._calc_A(t) - self._calc_B(t) * r0)

        sigma_fwd = self._calc_fwd_vol(t)
        return torch.exp(
            (r0 - self.b) * (torch.exp(-self.a * t) - 1) / self.a - \
            self.b * t + sigma_fwd ** 2 * t / (2 * self.a ** 2) + \
            sigma_fwd ** 2 * (4 * torch.exp(-self.a * t) - torch.exp(-2 * self.a * t) - 3) / (4 * self.a ** 3)
        )

    def calc_fwd(self, r0, t, delta):
        """F(0; t, t+delta) = 1/delta * (P(0,t) / P(0,t+delta) - 1)"""
        zcb_t = self.calc_zcb(r0, t)
        zcb_tdt = self.calc_zcb(r0, t + delta)
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

        K_bar = 1 / (1 + delta * K)
        vol_integral = self.sigma ** 2 / (2 * self.a ** 3) * (
                    1 - torch.exp(-2 * self.a * t) + torch.exp(-2 * self.a * delta) - \
                    torch.exp(-2 * self.a * (t + delta)) - 2 * (torch.exp(-self.a * delta) - \
                                                                torch.exp(-self.a * (2 * t + delta)))
                    )[:-1]
        d1 = (torch.log(zcb[:-1] * K_bar / zcb[1:]) + 0.5 * vol_integral) / torch.sqrt(vol_integral)
        d2 = d1 - torch.sqrt(vol_integral)

        return zcb[:-1] * N_cdf(d1) - zcb[1:] / K_bar * N_cdf(d2)

    def calc_cap(self, r0, t, delta, K):
        """Cp(0, t, t+delta) = sum_{i=1}^n Cpl(t; Ti_1, Ti) """
        return torch.sum(self.calc_cpl(r0, t, delta, K))

    def simulate(self, Z):
        """
        `Exact` simulation of the short rate, r(t) using
        Eq.(3.46), p. 110 - Glasserman (2003):
        """

        dt = self.timeline[1:] - self.timeline[:-1]

        self._x[0, :] = self.r0
        idx_zcb = 0
        idx_fwd = 0

        # Iterate over model's timeline
        for k, s in enumerate(self.timeline[1:]):
            self._x[k+1, :] = torch.exp(-self.a * dt[k]) * self._x[k, :] + self.b * (1 - torch.exp(-self.a * dt[k])) + \
                        self.sigma * Z[k, ] * torch.sqrt(1 / (2 * self.a) * (1 - torch.exp(-2 * self.a * dt[k])))

            if s in self.defline.zcbMats:
                self._zcb[idx_zcb, :] = self.calc_zcb(self._x[k, :], s)
                idx_zcb += 1

            if s in self.defline.fwdMats:
                self._fwd[idx_fwd, :] = self.calc_fwd(self._x[k, :], s, self.defline.fwdDeltas[idx_fwd])
                idx_fwd += 1

        return self._x, self._zcb, self._fwd

    def simulate_euler(self, r0, Z, dt):
        """
        p. 110 - Glasserman (2003)
        """
        # return r0 + self.a * (self.b - r0) * dt + self.sigma * torch.sqrt(dt) * Z
        raise NotImplementedError


def calibrate_vasicek(maturities, strikes, market_prices, a=1.00, b=0.05, sigma=0.2, r0=0.05, delta=0.25):
    def obj(x):
        x = torch.tensor(x)
        a, b, sigma, r0 = x[0], x[1], x[2], x[3]
        model = Vasicek(a, b, sigma)
        model_prices = torch.empty_like(market_prices, dtype=torch.float64)

        for i, T in enumerate(maturities):
            t = torch.linspace(start=delta, end=T, steps=int(T / delta))
            cap = model.calc_cap(r0, t, delta, strikes[i])
            model_prices[i] = cap

        err = model_prices - market_prices
        mse = torch.linalg.norm(err)**2
        return mse

    return scipy.optimize.minimize(
        fun=obj, x0=torch.tensor([a, b, sigma, r0]), method='Nelder-Mead', tol=1e-12,
        bounds=[(1E-6, 100.0), (-0.05, 1.00), (1E-6, 5.00), (-0.05, 1.00)],
        options={
            'xatol': 1e-12,
            'fatol': 1e-12,
            'maxiter': 2500,
            'maxfev': 2500,
            'adaptive': True,
            'disp': True
        })
