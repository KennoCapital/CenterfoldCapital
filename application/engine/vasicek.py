import torch
import scipy
from application.utils.torch_utils import N_cdf
from application.engine.mcBase import Model, MEASURES
from application.engine.products import SampleDef, Sample
from application.engine.linearProducts import forward, swap, swap_rate


class Vasicek(Model):
    """
        dr(t) = a*[b-r(t)]*dt + sigma*dW(t)
    """
    def __init__(self,
                 a,
                 b,
                 sigma,
                 r0=None,
                 use_ATS: bool = False,
                 measure: str = 'risk_neutral'):
        """
        :param a:               Mean reversion rate
        :param b:               Long term mean rate
        :param sigma:           Volatility,
        :param r0:              Initial value of instantaneous short rate
        :param measure:         Specifies which measure to simulate under
        :param use_ATS:         Use Affine Term Structure specification to calculate ZCB prices
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0
        self.measure = measure
        self.use_ATS = use_ATS

        if measure not in MEASURES:
            raise NotImplementedError(f'The measure "{measure}" is not implemented. '
                                      f'Use one of the following measures: {MEASURES}')

        # Attributes for Monte Carlo
        self._timeline = None
        self._defline = None
        self._x = None
        self._paths = None
        self._eulerTimeline = None
        self._tl_idx_mkt = None

    @property
    def timeline(self):
        return self._timeline

    @property
    def eulerTimeline(self):
        return self._eulerTimeline

    @property
    def defline(self):
        return self._defline

    @property
    def x(self):
        return self._x

    @property
    def paths(self):
        return self._paths

    def allocate(self,
                 prdTimeline:       torch.Tensor,
                 defline:           list[SampleDef],
                 N:                 int,
                 dTimeline:         torch.Tensor = torch.tensor([])):

        TL = [torch.tensor([0.0]), prdTimeline, dTimeline]
        self._eulerTimeline = dTimeline
        self._timeline = torch.unique(torch.concat(TL, dim=0), sorted=True)

        self._defline = defline

        # Allocate space for state and paths (market variables)
        n = len(prdTimeline)
        self._x = torch.full(size=(len(self.timeline), N), fill_value=torch.nan)
        self._paths = [
            Sample(
                fwd=[torch.full(size=(N, ), fill_value=torch.nan) for _ in range(len(defline[j].fwdRates))],
                irs=[torch.full(size=(N, ), fill_value=torch.nan) for _ in range(len(defline[j].irs))],
                disc=[torch.full(size=(N, ), fill_value=torch.nan) for _ in range(len(defline[j].discMats))],
                numeraire=torch.full(size=(N, ), fill_value=torch.nan) if defline[j].numeraire else None
            ) for j in range(n)
        ]

        # Specify indices of when to compute market variables
        self._tl_idx_mkt = [t in prdTimeline for t in self.timeline]

    def _exact_step(self, x, dt, Z):
        """
        Exact` simulation of the short rate, r(t), using
            Eq. (3.46), p. 110 - Glasserman (2003)
        """
        return torch.exp(-self.a * dt) * x + \
            self.b * (1 - torch.exp(-self.a * dt)) + \
            self.sigma * Z * torch.sqrt(1 / (2 * self.a) * (1 - torch.exp(-2 * self.a * dt)))

    def _euler_step(self, x, dt, Z):
        """
        Euler discretiation of the short rate, r(t), using
            p. 110 - Glasserman (2003)
        """
        return x + self.a * (self.b - x) * dt + self.sigma * torch.sqrt(dt) * Z

    def simulate(self, Z):
        # Decide function for performing simulation of state variable
        if len(self.eulerTimeline) > 0:
            step_func = self._euler_step
        else:
            step_func = self._exact_step

        # Calculate size of time steps
        dt = self.timeline[1:] - self.timeline[:-1]

        # Initialize state variables and set auxiliary index
        self._x[0, :] = self.r0
        idx = 0

        # Initialize numeraire
        numeraire = torch.ones_like(self.x[0, :])

        # Samples at time 0
        if self._tl_idx_mkt[0]:
            if self.measure == 'risk_neutral':
                for j in range(len(self.paths[idx].fwd)):
                    self._paths[idx].fwd[j] = self.calc_fwd(r0=self._x[0, :],
                                                            t=self.defline[idx].fwdRates[j].startDate,
                                                            delta=self.defline[idx].fwdRates[j].delta)

                for j in range(len(self.paths[idx].irs)):
                    self._paths[idx].irs[j] = self.calc_swap(r0=self._x[0, :],
                                                             t=self.defline[idx].irs[j].t,
                                                             delta=self.defline[idx].irs[j].delta,
                                                             K=self.defline[idx].irs[j].fixRate,
                                                             N=self.defline[idx].irs[j].notional)

                for j in range(len(self.paths[idx].disc)):
                    self._paths[idx].disc[j] = self.calc_zcb(r0=self._x[0, :],
                                                             t=self.defline[idx].discMats[j])

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire = numeraire

            idx += 1

        # Iterate over model's timeline
        for k, s in enumerate(self.timeline[1:]):
            # State variable
            self._x[k+1, :] = step_func(self.x[k, :], dt[k], Z[k, :])

            # Numeraire
            if self.measure == 'risk_neutral':
                # Trapezoidal rule: B(t) = exp{ int_0^t r(s) ds } ~ exp{sum[ r(t) * dt ]}
                numeraire *= torch.exp(0.5 * (self._x[k + 1, :] + self._x[k, :]) * dt[k])

            # Samples (market variables)
            if self._tl_idx_mkt[k + 1]:
                for j in range(len(self.paths[idx].fwd)):
                    self._paths[idx].fwd[j] = self.calc_fwd(r0=self._x[k + 1, :],
                                                            t=self.defline[idx].fwdRates[j].startDate - s,
                                                            delta=self.defline[idx].fwdRates[j].delta)

                for j in range(len(self.paths[idx].irs)):
                    self._paths[idx].irs[j] = self.calc_swap(r0=self._x[k + 1, :],
                                                             t=self.defline[idx].irs[j].t - s,
                                                             delta=self.defline[idx].irs[j].delta,
                                                             K=self.defline[idx].irs[j].fixRate,
                                                             N=self.defline[idx].irs[j].notional)

                for j in range(len(self.paths[idx].disc)):
                    self._paths[idx].disc[j] = self.calc_zcb(r0=self._x[k + 1, :],
                                                             t=self.defline[idx].discMats[j] - s)

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire = numeraire

                idx += 1



        return self.paths

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
        zcb_t = self.calc_zcb(r0, t)
        zcb_tdt = self.calc_zcb(r0, t + delta)
        return forward(zcb_t, zcb_tdt, delta)

    def calc_swap(self, r0, t, delta, K=None, N=torch.tensor(1.0)):
        """t = T_0, ..., T_n (future dates)"""
        zcb = self.calc_zcb(r0, t)
        return swap(zcb, delta, K, N)

    def calc_swap_rate(self, r0, t, delta):
        """t = T_0, ..., T_n (future dates)"""
        zcb = self.calc_zcb(r0, t)
        return swap_rate(zcb, delta)

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
