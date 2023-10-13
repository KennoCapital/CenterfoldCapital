import torch
import scipy
from application.utils.torch_utils import N_cdf
from application.engine.mcBase import Model, MEASURES
from application.engine.products import SampleDef, Sample
from application.engine.linearProducts import forward, swap, swap_rate


class trolleSchwartz(Model):
    """
        state variables:
        x, v, phi1, ph2, phi3, phi4, phi5, phi6
    """
    def __init__(self,
                 gamma,
                 kappa,
                 theta,
                 rho,
                 sigma,
                 alpha0,
                 alpha1,
                 x0, v0, phi1_0, phi2_0, phi3_0, phi4_0, phi5_0, phi6_0,
                 simDim: int = 1,
                 use_euler: bool = True,
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
        self.sigma = sigma
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

        if any( [i.size()[0] != simDim for i in [x0,v0,phi1_0,phi2_0,phi3_0,phi4_0, phi5_0, phi6_0] ] ):
            raise ValueError(f'Initial values must have dimension {simDim}')

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
        return [self._x, self._v, self._phi1, self._phi2, self._phi3, self._phi4, self._phi5, self._phi6]

    @property
    def W(self):
        return [self._Wf, self._Wv]

    @property
    def f(self):
        return self.calc_instant_fwd(X=self.x, t=0, T=self.timeline, f0=0.0)

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
        # 8 state vars: x, v, phi1, phi2, phi3, phi4, phi5, phi6
        self._x = torch.full(size=(self.simDim, len(self.timeline), N), fill_value=torch.nan)
        self._v = torch.clone(self._x)
        self._phi1 = torch.clone(self._x)
        self._phi2 = torch.clone(self._x)
        self._phi3 = torch.clone(self._x)
        self._phi4 = torch.clone(self._x)
        self._phi5 = torch.clone(self._x)
        self._phi6 = torch.clone(self._x)

        # for each event date we have a Sample object
        # which contains lists of fwdRates, irs, discMats, numeraire
        self._paths = [
            Sample(
                fwd=[torch.full(size=(N,), fill_value=torch.nan) for _ in range(len(defline[j].fwdRates))],
                irs=[torch.full(size=(N,), fill_value=torch.nan) for _ in range(len(defline[j].irs))],
                disc=[torch.full(size=(N,), fill_value=torch.nan) for _ in range(len(defline[j].discMats))],
                numeraire=torch.full(size=(N,), fill_value=torch.nan) if defline[j].numeraire else None
            ) for j in range(n)
        ]

        # Specify indices of when to compute market variables
        self._tl_idx_mkt = [t in prdTimeline for t in self.timeline]

    def _correlatedBrownians(self, Z):
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
        L = torch.linalg.cholesky(covMat)
        # Correlated BMs: W = Z x L^T
        W = Z.reshape(-1,2) @ L.t()

        Wf = W[:,0]#torch.cumsum(W[:,0], dim=0)
        Wv = W[:,1]#torch.cumsum(W[:,1], dim=0)

        Wf = Wf.reshape((len(self.timeline)-1,-1))
        Wv = Wv.reshape((len(self.timeline)-1,-1))

        self._Wf = Wf
        self._Wv = Wv

        return Wf, Wv

    def _sigma(self, t, T):
        """sigma(0,t) = (alpha0 + alpha1(T-t)) * exp^{ -gamma *(T-t) }"""
        return (self.alpha0 + self.alpha1 * (T-t)) * torch.exp(-self.gamma * (T-t))

    def _euler_step(self, x, v, phi1, phi2, phi3, phi4, phi5, phi6, dt, dWf, dWv):
        """
        Euler discretisation of the state variables:
        x, v, phi1, phi2, phi3, phi4, phi5, phi6
        """
        dx = -self.gamma * x * dt + torch.sqrt(v) * dWf * torch.sqrt(dt)

        #dv = self.kappa * (self.theta - v) * dt + self._sigma(t=0, T=dt) * torch.sqrt(v) * dWv * torch.sqrt(dt)
        dv = self.kappa * (self.theta - v) * dt + self.sigma * torch.sqrt(v) * dWv * torch.sqrt(dt)

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

        return x,v,phi1,phi2,phi3,phi4,phi5,phi6

    def simulate(self, Z):
        # Decide function for performing simulation of state variable
        step_func = self._euler_step

        # Calculate size of time steps
        dt = self.timeline[1:] - self.timeline[:-1]

        # Compute correlated Brownian motions
        dWf, dWv = self._correlatedBrownians(Z)

        # Initialize state variables
        # set initial values to same for all paths
        self._x[:, 0, :] = self.x0
        self._v[:, 0, :] = self.v0
        self._phi1[:, 0, :] = self.phi1_0
        self._phi2[:, 0, :] = self.phi2_0
        self._phi3[:, 0, :] = self.phi3_0
        self._phi4[:, 0, :] = self.phi4_0
        self._phi5[:, 0, :] = self.phi5_0
        self._phi6[:, 0, :] = self.phi6_0

        # and set auxiliary index
        idx = 0
        s = 0.0

        # Initialize numeraire
        numeraire = torch.ones_like(self._x[0, 0, :])

        def _fillSample(x):
            nonlocal idx, s, numeraire
            if self.measure == 'risk_neutral':
                for j in range(len(self.paths[idx].fwd)):
                    self._paths[idx].fwd[j] = self.calc_fwd(X=x,
                                                            t=self.defline[idx].fwdRates[j].startDate - s,
                                                            delta=self.defline[idx].fwdRates[j].delta)

                for j in range(len(self.paths[idx].irs)):
                    self._paths[idx].irs[j] = self.calc_swap(X=x,
                                                             t=self.defline[idx].irs[j].t - s,
                                                             delta=self.defline[idx].irs[j].delta,
                                                             K=self.defline[idx].irs[j].fixRate,
                                                             N=self.defline[idx].irs[j].notional)

                for j in range(len(self.paths[idx].disc)):
                    self._paths[idx].disc[j] = self.calc_zcb(PT= 1.0,#todo: figure out this quantity
                                                             Pt = 1.0,
                                                             X=x,
                                                             t=0,
                                                             T=self.defline[idx].discMats[j] - s)

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire[:] = numeraire

            idx += 1

        # Samples at time 0
        if self._tl_idx_mkt[0]:
            _fillSample(x=[self._x[:, 0, :], self._v[:, 0, :],
                           self._phi1[:, 0, :], self._phi2[:, 0, :],
                           self._phi3[:, 0, :], self._phi4[:, 0, :],
                           self._phi5[:, 0, :], self._phi6[:, 0, :]])

        # Iterate over model's timeline
        for k, s in enumerate(self.timeline[1:]):

            # State variables using step function
            self._x[:,k+1, :], self._v[:,k+1, :], self._phi1[:,k+1, :], self._phi2[:,k+1, :], \
            self._phi3[:,k+1, :], self._phi4[:,k+1, :], self._phi5[:,k+1, :], self._phi6[:,k+1, :] = \
                step_func(self._x[:,k, :], self._v[:,k, :], self._phi1[:,k, :], self._phi2[:,k, :],
                          self._phi3[:,k, :], self._phi4[:,k, :], self._phi5[:,k, :], self._phi6[:,k, :],
                          dt[k], dWf[k,:], dWv[k, :])

            # Numeraire
            if self.measure == 'risk_neutral':
                # Trapezoidal rule: B(t) = exp{ int_0^t r(s) ds } ~ exp{sum[ r(t) * dt ]}
                rt1 = self.calc_short_rate(X=[self._x[:,k+1, :], self._v[:,k+1, :], self._phi1[:,k+1, :], self._phi2[:,k+1, :], \
            self._phi3[:,k+1, :], self._phi4[:,k+1, :], self._phi5[:,k+1, :], self._phi6[:,k+1, :]], t=self.timeline[k+1])
                rt =self.calc_short_rate(X=[self._x[:,k, :], self._v[:,k, :], self._phi1[:,k, :], self._phi2[:,k, :], \
            self._phi3[:,k, :], self._phi4[:,k, :], self._phi5[:,k, :], self._phi6[:,k, :]], t=self.timeline[k])
                numeraire *= torch.exp(0.5 * (rt1 + rt) * dt[k])

                #numeraire *= 1.0 torch.exp(0.5 * (self._x[k + 1, :] + self._x[k, :]) * dt[k])

            # Samples (market variables)
            if self._tl_idx_mkt[k + 1]:
                _fillSample(x=[self._x[:, k+1, :], self._v[:, k+1, :],
                    self._phi1[:, k+1, :], self._phi2[:, k+1, :],
                    self._phi3[:, k+1, :], self._phi4[:, k+1, :],
                    self._phi5[:, k+1, :], self._phi6[:, k+1, :]]
                            )
        return self.paths

    def calc_zcb(self, PT, Pt, X, t, T):
        """
        # todo: not sure how to set PT (and Pt, unless we have Pt=P(0,0)= 1.0)
        P(t, T)=P(0,T)/P(0,t) * exp{sum_i(Bx_i(T-t)x_i(t)) + sum_i(sum_j(B_phi_{j,i}(T-t) * phi_{j,i}(t)))}

        Note:
        X = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
        """
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i for i in X]

        Bx = self.alpha1 / self.gamma * ( (1/self.gamma + self.alpha0/self.alpha1) * (torch.exp(-self.gamma*(T-t)) - 1) + \
                                          (T-t) * torch.exp(-self.gamma*(T-t)) )
        Bphi1 = self.alpha1 / self.gamma * ( torch.exp(-self.gamma*(T-t)) - 1 )
        Bphi2 = torch.pow(self.alpha1 / self.gamma, 2) * (1/self.gamma + self.alpha0/self.alpha1) *\
                ( (1/self.gamma + self.alpha0/self.alpha1) * (torch.exp(-self.gamma*(T-t)) - 1) + \
                  (T-t) * torch.exp(-self.gamma*(T-t)) )
        Bphi3 = - self.alpha1 / torch.pow(self.gamma, 2) * ( (self.alpha1 / (2*torch.pow(self.gamma,2)) + self.alpha0 / self.gamma +\
                                                              torch.pow(self.alpha0,2) / (2*self.alpha1) ) * (torch.exp(-2*self.gamma*(T-t)) - 1) +\
                                                             (self.alpha1 / self.gamma + self.alpha0) * (T-t) * torch.exp(-2*self.gamma*(T-t)) +\
                                                             self.alpha1 / 2 * (T-t)**2 * torch.exp(-2*self.gamma*(T-t)) )
        Bphi4 = torch.pow(self.alpha1 / self.gamma,2) * (1/self.gamma + self.alpha0/self.alpha1) * (torch.exp(-self.gamma*(T-t)) - 1)
        Bphi5 = - self.alpha1 / torch.pow(self.gamma,2) * ( (self.alpha1 / self.gamma + self.alpha0) * \
                                                            (torch.exp(-2*self.gamma*(T-t)) -1) + self.alpha1 * (T-t) * torch.exp(-2*self.gamma*(T-t)) )
        Bphi6 = - 0.5 * torch.pow(self.alpha1 / self.gamma,2) * (torch.exp(-2*self.gamma*(T-t)) - 1)

        # sum_i(Bx_i(T-t)x_i(t))
        Bx_sum = torch.sum(Bx * x,dim=0)

        # sum_i(sum_j(B_phi_{j,i}(T-t) * phi_{j,i}(t)))
        Bphi_sum = torch.sum(Bphi1 * phi1 + Bphi2 * phi2 + Bphi3 * phi3 + Bphi4 * phi4 + Bphi5 * phi5 + Bphi6 * phi6, dim=0)

        return PT / Pt * torch.exp(Bx_sum + Bphi_sum)

    def calc_instant_fwd(self, X, t, T, f0=0.0):

        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i for i in X]

        Bx = (self.alpha0 + self.alpha1 * (T - t)) * torch.exp(-self.gamma * (T - t))
        Bphi1 = self.alpha1 * torch.exp(-self.gamma * (T - t))
        Bphi2 = self.alpha1 / self.gamma * (1/self.gamma + self.alpha0/self.alpha1) * \
                (self.alpha0 + self.alpha1*(T-t)) * torch.exp(-self.gamma * (T - t))
        Bphi3 = - ( self.alpha0 * self.alpha1 / self.gamma * (1/self.gamma + self.alpha0/self.alpha1) +\
                    self.alpha1/self.gamma * (self.alpha1 / self.gamma + 2 * self.alpha0)*(T-t) + \
                    torch.pow(self.alpha1,2) /self.gamma * (T-t)**2 ) * torch.exp(-2 * self.gamma * (T - t))
        Bphi4 = torch.pow(self.alpha1,2) / self.gamma * (1/self.gamma + self.alpha0/self.alpha1) * torch.exp(-self.gamma * (T - t))
        Bphi5 = - self.alpha1 / self.gamma * (self.alpha1 / self.gamma + 2*self.alpha0 + 2* self.alpha1*(T-t)) * torch.exp(-2 * self.gamma * (T - t))
        Bphi6 = - torch.pow(self.alpha1,2) / self.gamma * torch.exp(-2 * self.gamma * (T - t))

        # sum_i(Bx_i(T-t)x_i(t))
        Bx_sum = torch.sum(Bx * x, dim=0)

        # sum_i(sum_j(B_phi_{j,i}(T-t) * phi_{j,i}(t)))
        Bphi_sum = torch.sum(Bphi1 * phi1 + Bphi2 * phi2 + Bphi3 * phi3 + Bphi4 * phi4 + Bphi5 * phi5 + Bphi6 * phi6, dim=0)

        return f0 + Bx_sum + Bphi_sum

    def calc_short_rate(self, X, t, f0=0.0):
        return self.calc_instant_fwd(X, t, t, f0)

    def calc_fwd(self, X, t, delta):
        # todo: not sure how to set PT and Pt
        zcb_t = self.calc_zcb(PT=0.98, Pt=1.0, X=X, t=0, T=t)
        zcb_tdt = self.calc_zcb(PT=0.98, Pt=1.0, X=X, t=0, T=t+delta)

        return forward(zcb_t, zcb_tdt, delta)

    def calc_swap(self, X, t, delta, K=None, N=torch.tensor(1.0)):
        """t = T_0, ..., T_n (future dates)"""
        # todo: not sure how to set PT and Pt
        zcb = self.calc_zcb(PT=1.0, Pt=1.0, X=X, t=0, T=t)
        return swap(zcb, delta, K, N)

    def calc_swap_rate(self, X, t, delta):
        """t = T_0, ..., T_n (future dates)"""
        # todo: not sure how to set PT and Pt
        zcb = self.calc_zcb(PT=1.0, Pt=1.0, X=X, t=0, T=t)
        return swap_rate(zcb, delta)

    def calc_cpl(self, X, t, delta, K): # todo rewrite this method
        """
           Revise Trolle-Schwartz on this
                Cpl(0; t, t+delta) = P(0,t) * N(d1) - P(0,t+delta) / K_bar * N(d2)
        """
        zcb = self.calc_zcb(X, t)

        K_bar = 1 / (1 + delta * K)
        vol_integral = self.sigma ** 2 / (2 * self.a ** 3) * (
                    1 - torch.exp(-2 * self.a * t) + torch.exp(-2 * self.a * delta) - \
                    torch.exp(-2 * self.a * (t + delta)) - 2 * (torch.exp(-self.a * delta) - \
                                                                torch.exp(-self.a * (2 * t + delta)))
                    )[:-1]
        d1 = (torch.log(zcb[:-1] * K_bar / zcb[1:]) + 0.5 * vol_integral) / torch.sqrt(vol_integral)
        d2 = d1 - torch.sqrt(vol_integral)

        return torch.tensor(1.0) # #zcb[:-1] * N_cdf(d1) - zcb[1:] / K_bar * N_cdf(d2)

    def calc_cap(self, x, t, delta, K):
        """Cp(0, t, t+delta) = sum_{i=1}^n Cpl(t; Ti_1, Ti) """
        return torch.tensor(1.0) #torch.sum(self.calc_cpl(x, t, delta, K))
