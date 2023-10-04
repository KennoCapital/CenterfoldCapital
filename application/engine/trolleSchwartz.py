import torch
import scipy
from application.utils.torch_utils import N_cdf
from application.engine.mcBase import Model, MEASURES
from application.engine.products import SampleDef, Sample
from application.engine.linearProducts import forward, swap, swap_rate


class trolleSchwartz(Model):
    """
        state variables:
        dx, dv, dphi1, dph2, dphi3, dphi4, dphi5, dphi6
    """
    def __init__(self,
                 gamma,
                 kappa,
                 theta,
                 rho,
                 alpha0,
                 alpha1,
                 x0, v0, phi1_0, phi2_0, phi3_0, phi4_0, phi5_0, phi6_0,
                 simDim: int = 1,
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
        self.gamma = gamma
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.x0 = x0
        self.v0 = v0
        self.phi1_0 = phi1_0
        self.phi2_0 = phi2_0
        self.phi3_0 = phi3_0
        self.phi4_0 = phi4_0
        self.phi5_0 = phi5_0
        self.phi6_0 = phi6_0
        self.simDim = simDim

        self.use_euler = use_euler
        self.measure = measure

        if measure not in MEASURES:
            raise NotImplementedError(f'The measure "{measure}" is not implemented. '
                                      f'Use one of the following measures: {MEASURES}')

        if any( [i.size()[0] != dim for i in [x0,v0,phi1_0,phi2_0,phi3_0,phi4_0, phi5_0, phi6_0] ] ):
            raise ValueError(f'Initial values must have dimension {dim}')

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
        if self.simDim > 1:
            self._x = torch.full(size=(len(self.timeline), self.simDim, N), fill_value=torch.nan)
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

    def correlatedBrownians(self, Z):
        """ Generate correlated Brownian motions """
        # Covariance matrix
        if self.simDim > 1:
            # we would need a cov matrix for each simulation dimension
            covMat = torch.empty( (self.simDim, 2, 2))

            # but we only have two sources of randomness
            for i in range(len(self.rho)):
                covMat[i, 0, 1] = self.rho[i]
                covMat[i, 1, 0] = self.rho[i]
        else:
            covMat = torch.tensor([[1.0, self.rho],
                                   [self.rho, 1.0]])

        # Cholesky decomposition: covMat = L x L^T
        L = torch.linalg.cholesky_ex(covMat)
        # Correlated BMs: W = Z x L^T
        W = Z @ L.t() #todo: consider reshaping Z

        return W

    def sigma(self, tau):
        """sigma(0,t) = (alpha0 + alpha1(T-t)) * exp^{ -gamma *(T-t) }"""
        return (self.alpha0 + self.alpha1 * (tau)) * torch.exp(-self.gamma * (tau))

    def _euler_step(self, x, v, phi1, phi2, phi3, phi4, phi5, phi6, dt, dWf, dWv):
        """
        Euler discretisation of the state variables
        """

        dx = -self.gamma * x * dt + torch.sqrt(v) * dWf
        dv = self.kappa * (self.theta - v) * dt + sigma() * torch.sqrt(v) * dWv

        dphi1 = (x - self.gamma * phi1) * dt
        dphi2 = (v - self.gamma * phi2) * dt
        dphi3 = (v - 2 * self.gamma * phi3) * dt
        dphi4 = (phi2 - self.gamma * phi4) * dt
        dphi5 = (phi3 - 2 * self.gamma * phi5) * dt
        dphi6 = (2 * phi5 - 2 * self.gamma * phi6) * dt

        x += dx
        v += dv
        phi1 += dphi1
        phi2 += dphi2
        phi3 += dphi3
        phi4 += dphi4
        phi5 += dphi5
        phi6 += dphi6

        return [x,v,phi1,phi2,phi3,phi4,phi5,phi6]

    def simulate(self, Z):
        # Decide function for performing simulation of state variable
        step_func = self._euler_step

        # Calculate size of time steps
        dt = self.timeline[1:] - self.timeline[:-1]

        # Initialize state variables and set auxiliary index
        if self.simDim > 1:
            for i in range(self.simDim):
                self._x[i,:,0, :] = torch.tensor([[self.x0[i],
                                   self.v0[i],
                                   self.phi1_0[i], self.phi2_0[i], self.phi3_0[i],
                                   self.phi4_0[i], self.phi5_0[i], self.phi6_0[i]] for _ in range(8)])
        else:
            self._x[:, 0, :] = torch.tensor([[self.x0,
                                                 self.v0,
                                                 self.phi1_0, self.phi2_0, self.phi3_0,
                                                 self.phi4_0, self.phi5_0, self.phi6_0] for _ in range(8)])

        idx = 0

        # Initialize numeraire
        numeraire = torch.ones_like(self.x[0, :])

        def _fillSample(x):
            nonlocal idx, s, numeraire
            if self.measure == 'risk_neutral':
                for j in range(len(self.paths[idx].fwd)):
                    self._paths[idx].fwd[j] = self.calc_fwd(r0=x,
                                                            t=self.defline[idx].fwdRates[j].startDate - s,
                                                            delta=self.defline[idx].fwdRates[j].delta)

                for j in range(len(self.paths[idx].irs)):
                    self._paths[idx].irs[j] = self.calc_swap(r0=x,
                                                             t=self.defline[idx].irs[j].t - s,
                                                             delta=self.defline[idx].irs[j].delta,
                                                             K=self.defline[idx].irs[j].fixRate,
                                                             N=self.defline[idx].irs[j].notional)

                for j in range(len(self.paths[idx].disc)):
                    self._paths[idx].disc[j] = self.calc_zcb(r0=x,
                                                             t=self.defline[idx].discMats[j] - s)

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire[:] = numeraire

            idx += 1

        # Samples at time 0
        if self._tl_idx_mkt[0]:
            _fillSample(x=self._x[0, :])

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


