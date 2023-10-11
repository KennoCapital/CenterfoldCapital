import torch
import scipy
from application.utils.torch_utils import N_cdf
from application.engine.mcBase import Model, MEASURES
from application.engine.products import Product, Sample
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
                 use_ATS:   bool = False,
                 use_euler: bool = False,
                 measure:   str = 'risk_neutral'):
        """
        :param a:               Mean reversion rate
        :param b:               Long term mean rate
        :param sigma:           Volatility,
        :param r0:              Initial value of instantaneous short rate
        :param measure:         Specifies which measure to simulate under
        :param use_euler:       Use Euler discretization (True) or Exact method (False) for simulation of short rate
        :param use_ATS:         Use Affine Term Structure specification to calculate ZCB prices
        """
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0
        self.measure = measure
        self.use_euler = use_euler
        self.use_ATS = use_ATS

        if measure not in MEASURES:
            raise NotImplementedError(f'The measure "{measure}" is not implemented. '
                                      f'Use one of the following measures: {MEASURES}')

        # Attributes for Monte Carlo
        self._timeline = None
        self._defline = None
        self._x = None
        self._paths = None
        self._dTimeline = None
        self._tl_idx_mkt = None
        self._Tn = None

    @property
    def timeline(self):
        return self._timeline

    @property
    def dTimeline(self):
        return self._dTimeline

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
                 prd:               Product,
                 N:                 int,
                 dTimeline:         torch.Tensor = torch.tensor([])):

        TL = [torch.tensor([0.0]), prd.timeline, dTimeline]
        self._dTimeline = dTimeline
        self._timeline = torch.unique(torch.concat(TL, dim=0), sorted=True)
        self._defline = prd.defline

        self._Tn = prd.Tn

        # Allocate space for state and paths (market variables)
        self._x = torch.full(size=(len(self.timeline), N), fill_value=torch.nan)
        self._paths = [
            Sample(
                fwd=[torch.full(size=(N, ), fill_value=torch.nan) for _ in range(len(prd.defline[j].fwdRates))],
                irs=[torch.full(size=(N, ), fill_value=torch.nan) for _ in range(len(prd.defline[j].irs))],
                disc=[torch.full(size=(N, ), fill_value=torch.nan) for _ in range(len(prd.defline[j].discMats))],
                numeraire=torch.full(size=(N, ), fill_value=torch.nan) if prd.defline[j].numeraire else None,
                x=torch.full(size=(N, ), fill_value=torch.nan) if prd.defline[j].stateVar else None
            ) for j in range(len(prd.timeline))
        ]

        # Specify indices of when to compute market variables
        self._tl_idx_mkt = [t in prd.timeline for t in self.timeline]

    def _exact_step(self, x, dt, Z, s):
        """
        Exact` simulation of the short rate, r(t), using
            Eq. (3.46), p. 110 - Glasserman (2003)
        """
        if self.measure == 'risk_neutral':
            return torch.exp(-self.a * dt) * x + \
                self.b * (1 - torch.exp(-self.a * dt)) + \
                self.sigma * Z * torch.sqrt(1 / (2 * self.a) * (1 - torch.exp(-2 * self.a * dt)))

        if self.measure == 'terminal':
            return torch.exp(-self.a * dt) * x + \
                (self.b - self.sigma ** 2 * self._calc_B(self._Tn-s)) * (1 - torch.exp(-self.a * dt)) + \
                self.sigma * Z * torch.sqrt(1 / (2 * self.a) * (1 - torch.exp(-2 * self.a * dt)))

    def _euler_step(self, x, dt, Z, s):
        """
        Euler discretiation of the short rate, r(t), using
            p. 110 - Glasserman (2003)
        """
        if self.measure == 'risk_neutral':
            return x + self.a * (self.b - x) * dt + self.sigma * torch.sqrt(dt) * Z

        if self.measure == 'terminal':
            return x + self.a * ((self.b - self.sigma ** 2 * self._calc_B(self._Tn-s)) - x) * dt + \
                self.sigma * torch.sqrt(dt) * Z

    def simulate(self, Z):
        # Decide function for performing simulation of state variable
        if self.use_euler:
            step_func = self._euler_step
        else:
            step_func = self._exact_step

        # Calculate size of time steps
        dt = self.timeline[1:] - self.timeline[:-1]

        # Set auxiliary index for samples and auxiliary variable for calculating the integral of the short rate
        idx = 0
        s = 0
        sum_x = torch.zeros_like(self.x[0, :])

        # Initialize state variable
        self._x[0, :] = self.r0

        def _fillSample(x):
            nonlocal idx, s, sum_x
            if self.measure == 'risk_neutral':
                for j in range(len(self.paths[idx].fwd)):
                    self._paths[idx].fwd[j][:] = self.calc_fwd(r0=x,
                                                               t=self.defline[idx].fwdRates[j].startDate - s,
                                                               delta=self.defline[idx].fwdRates[j].delta)

                for j in range(len(self.paths[idx].irs)):
                    self._paths[idx].irs[j][:] = self.calc_swap(r0=x,
                                                                t=self.defline[idx].irs[j].t - s,
                                                                delta=self.defline[idx].irs[j].delta,
                                                                K=self.defline[idx].irs[j].fixRate,
                                                                N=self.defline[idx].irs[j].notional)

                for j in range(len(self.paths[idx].disc)):
                    self._paths[idx].disc[j][:] = self.calc_zcb(r0=x,
                                                                t=self.defline[idx].discMats[j] - s)

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire[:] = torch.exp(sum_x)

                if self.paths[idx].x is not None:
                    self._paths[idx].x[:] = x

            if self.measure == 'terminal':
                for j in range(len(self.paths[idx].fwd)):
                    self._paths[idx].fwd[j][:] = self.calc_fwd(r0=x,
                                                               t=self.defline[idx].fwdRates[j].startDate - s,
                                                               delta=self.defline[idx].fwdRates[j].delta)

                for j in range(len(self.paths[idx].irs)):
                    self._paths[idx].irs[j][:] = self.calc_swap(r0=x,
                                                                t=self.defline[idx].irs[j].t - s,
                                                                delta=self.defline[idx].irs[j].delta,
                                                                K=self.defline[idx].irs[j].fixRate,
                                                                N=self.defline[idx].irs[j].notional)

                for j in range(len(self.paths[idx].disc)):
                    self._paths[idx].disc[j][:] = self.calc_zcb(r0=x,
                                                                t=self.defline[idx].discMats[j] - s)

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire[:] = self.calc_zcb(r0=x,
                                                                  t=self._Tn - s)

                if self.paths[idx].x is not None:
                    self._paths[idx].x[:] = x

            idx += 1

        # Samples at time 0
        if self._tl_idx_mkt[0]:
            _fillSample(x=self._x[0, :])

        # Iterate over model's timeline
        for k, s in enumerate(self.timeline[1:]):
            # State variable
            self._x[k+1, :] = step_func(self.x[k, :], dt[k], Z[k, :], s)

            if self.measure == 'risk_neutral':
                # Trapezoidal rule: int_0^t r(s) ds ~ sum[ 0.5 * (r(t+dt)-r(t)) * dt ]
                sum_x += 0.5 * (self._x[k + 1, :] + self._x[k, :]) * dt[k]

            # Samples (market variables)
            if self._tl_idx_mkt[k + 1]:
                _fillSample(x=self._x[k + 1, :])

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
        """t = T_0, ..., T_{n-1} (fixing dates)"""
        zcb = self.calc_zcb(r0, t)
        zcb_tdt = self.calc_zcb(r0, t[-1] + delta).view(-1)
        return swap(zcb, delta, K, N)

    def calc_swap_rate(self, r0, t, delta):
        """t = T_0, ..., T_{n-1} (fixing dates)"""
        zcb = self.calc_zcb(r0, t)
        zcb_tdt = self.calc_zcb(r0, t[-1] + delta).view(-1)
        return swap_rate(torch.concat([zcb, zcb_tdt]), delta)

    def calc_cpl(self, r0, t, delta, K):
        """
           Solution to Filipovic's prop. 7.2
                Cpl(0; t, t+delta) = P(0,t) * N(d1) - P(0,t+delta) / K_bar * N(d2)
        """
        # TODO rewrite this function so that t only needs to be t and not [t, t+delta]
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


def calibrate_vasicek_cap(maturities, strikes, market_prices, a=1.00, b=0.05, sigma=0.2, r0=0.05, delta=0.25):
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
        bounds=[(1E-12, 100.0), (-0.05, 1.00), (1E-12, 5.00), (-0.05, 1.00)],
        options={
            'xatol': 1e-12,
            'fatol': 1e-12,
            'maxiter': 2500,
            'maxfev': 2500,
            'adaptive': True,
            'disp': True
        })


def calibrate_vasicek_zcb_price(maturities, market_prices, a=1.00, b=0.05, sigma=0.2, r0=0.05):
    def obj(x):
        x = torch.tensor(x)
        a, b, sigma, r0 = x[0], x[1], x[2], x[3]
        model = Vasicek(a, b, sigma)
        model_prices = torch.empty_like(market_prices, dtype=torch.float64)

        for i, T in enumerate(maturities):
            zcb = model.calc_zcb(r0, T)
            model_prices[i] = zcb

        err = model_prices - market_prices
        mse = torch.linalg.norm(err)**2
        return mse

    return scipy.optimize.minimize(
        fun=obj, x0=torch.tensor([a, b, sigma, r0]), method='Nelder-Mead', tol=1e-12,
        bounds=[(1E-12, 1.000), (-0.05, 1.00), (1E-12, 5.00), (-0.05, 1.00)],
        options={
            'xatol': 1e-12,
            'fatol': 1e-12,
            'maxiter': 2500,
            'maxfev': 2500,
            'adaptive': True,
            'disp': True
        })
