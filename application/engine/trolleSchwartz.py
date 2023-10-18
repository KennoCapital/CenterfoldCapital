import torch
from application.engine.mcBase import Model, MEASURES
from application.engine.products import Product, Sample
from application.engine.linearProducts import forward, swap, swap_rate


class trolleSchwartz(Model):
    """
        Todo: add semi-analytic caplet pricing formula

        MC simulation of state variables:
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
                 r0 = 0.0,
                 simDim: int = 1,
                 numRV: int = 2,
                 use_euler: bool = True,
                 measure:   str = 'risk_neutral'):
        """
        :param gamma:           Parameter originating from HJM  volatility structure

        :param kappa:           Mean reversion rate variance process
        :param theta:           Long term mean level variance process
        :param rho:             Correlation between stochastic processes
        :param sigma:           Volatility parameter variance process,

        :param alpha0:          Parameter originating from HJM  volatility structure
        :param alpha1:          Parameter originating from HJM  volatility structure

        :param r0:              Initial value of instantaneous short rate
        :param simDim:          Multi-factor dimension

        :param use_euler:       [Supports currently only 'True'] Use Euler discretization (True) or Exact method (False) for simulation of short rate
        :param measure:         [Supports currently only 'risk_neutral'] Specifies which measure to simulate under
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

        self.r0 = r0
        self.simDim = simDim
        self._numRV = numRV

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
    def numRV(self):
        return self._numRV

    @property
    def x(self):
        return [self._x, self._v, self._phi1, self._phi2, self._phi3, self._phi4, self._phi5, self._phi6]

    @property
    def f(self):
        return self.calc_instant_fwd(X=self.x, t=0, T=self.timeline)

    @property
    def paths(self):
        return self._paths

    def allocate(self,
                 prd:       Product,
                 N:         int,
                 dTimeline: torch.tensor = torch.tensor([])):

        TL = [torch.tensor([0.0]), prd.timeline, dTimeline]
        self._dTimeline = dTimeline
        self._timeline = torch.unique(torch.concat(TL, dim=0), sorted=True)
        self._defline = prd.defline


        # Allocate space for state and paths (market variables)
        self._Tn = prd.Tn
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
                fwd=[torch.full(size=(N,), fill_value=torch.nan) for _ in range(len(prd.defline[j].fwdRates))],
                irs=[torch.full(size=(N,), fill_value=torch.nan) for _ in range(len(prd.defline[j].irs))],
                disc=[torch.full(size=(N,), fill_value=torch.nan) for _ in range(len(prd.defline[j].discMats))],
                numeraire=torch.full(size=(N,), fill_value=torch.nan) if prd.defline[j].numeraire else None,
                x=torch.full(size=(N,), fill_value=torch.nan) if prd.defline[j].stateVar else None
            ) for j in range(len(prd.timeline))
        ]

        # Specify indices of when to compute market variables
        self._tl_idx_mkt = [t in prd.timeline for t in self.timeline]

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

        Wf = W[:,0]
        Wv = W[:,1]

        Wf = Wf.reshape((len(self.timeline)-1,-1))
        Wv = Wv.reshape((len(self.timeline)-1,-1))

        return Wf, Wv

    def _sigma(self, t, T):
        """sigma(0,t) = (alpha0 + alpha1(T-t)) * exp^{ -gamma *(T-t) }"""
        return (self.alpha0 + self.alpha1 * (T-t)) * torch.exp(-self.gamma * (T-t))

    def _euler_step(self, x, v, phi1, phi2, phi3, phi4, phi5, phi6, dt, dWf, dWv):
        """
        Euler discretisation of the state variables:
        x, v, phi1, phi2, phi3, phi4, phi5, phi6
        """
        dx = -self.gamma * x * dt + torch.sqrt(torch.abs(v)) * dWf * torch.sqrt(dt)

        # Note: using abs(v)
        dv = self.kappa * (self.theta - v) * dt + self.sigma * torch.sqrt(torch.abs(v)) * dWv * torch.sqrt(dt)

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
                    self._paths[idx].disc[j] = self.calc_zcb_price(X=x,
                                                             t=0,
                                                             T=self.defline[idx].discMats[j] - s)

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire[:] = numeraire

                if self.paths[idx].x is not None:
                    self._paths[idx].x[:] = x

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
                xk1 = [self._x[:,k+1, :], self._v[:,k+1, :], self._phi1[:,k+1, :], self._phi2[:,k+1, :], \
                        self._phi3[:,k+1, :], self._phi4[:,k+1, :], self._phi5[:,k+1, :], self._phi6[:,k+1, :]]
                xk = [self._x[:,k, :], self._v[:,k, :], self._phi1[:,k, :], self._phi2[:,k, :], \
                        self._phi3[:,k, :], self._phi4[:,k, :], self._phi5[:,k, :], self._phi6[:,k, :]]

                rt1 = self.calc_short_rate(X=xk1, t=self.timeline[k+1])
                rt = self.calc_short_rate(X=xk, t=self.timeline[k])

                numeraire *= torch.exp(0.5 * (rt1 + rt) * dt[k])

            # Samples (market variables)
            if self._tl_idx_mkt[k + 1]:
                _fillSample(x=[self._x[:, k+1, :], self._v[:, k+1, :],
                    self._phi1[:, k+1, :], self._phi2[:, k+1, :],
                    self._phi3[:, k+1, :], self._phi4[:, k+1, :],
                    self._phi5[:, k+1, :], self._phi6[:, k+1, :]]
                            )
        return self.paths

    def calc_zcb(self,X, t, T,  PT=1.0, Pt=1.0):
        """
        P(t, T)=P(0,T)/P(0,t) * exp{sum_i(Bx_i(T-t)x_i(t)) + sum_i(sum_j(B_phi_{j,i}(T-t) * phi_{j,i}(t)))}

        Note:
        X = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
        """
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i for i in X]

        Bx = self.alpha1 / self.gamma * (
                    (1 / self.gamma + self.alpha0 / self.alpha1) * (torch.exp(-self.gamma * (T - t)) - 1) + \
                    (T - t) * torch.exp(-self.gamma * (T - t)))

        Bphi1 = self.alpha1 / self.gamma * (torch.exp(-self.gamma * (T - t)) - 1)
        Bphi2 = torch.pow(self.alpha1 / self.gamma, 2) * (1 / self.gamma + self.alpha0 / self.alpha1) * \
                ((1 / self.gamma + self.alpha0 / self.alpha1) * (torch.exp(-self.gamma * (T - t)) - 1) + \
                 (T - t) * torch.exp(-self.gamma * (T - t)))
        Bphi3 = - self.alpha1 / torch.pow(self.gamma, 2) * (
                    (self.alpha1 / (2 * torch.pow(self.gamma, 2)) + self.alpha0 / self.gamma + \
                     torch.pow(self.alpha0, 2) / (2 * self.alpha1)) * (torch.exp(-2 * self.gamma * (T - t)) - 1) + \
                    (self.alpha1 / self.gamma + self.alpha0) * (T - t) * torch.exp(-2 * self.gamma * (T - t)) + \
                    self.alpha1 / 2 * (T - t) ** 2 * torch.exp(-2 * self.gamma * (T - t)))
        Bphi4 = torch.pow(self.alpha1 / self.gamma, 2) * (1 / self.gamma + self.alpha0 / self.alpha1) * (
                    torch.exp(-self.gamma * (T - t)) - 1)
        Bphi5 = - self.alpha1 / torch.pow(self.gamma, 2) * ((self.alpha1 / self.gamma + self.alpha0) * \
                                                            (torch.exp(-2 * self.gamma * (T - t)) - 1) + self.alpha1 * (
                                                                        T - t) * torch.exp(-2 * self.gamma * (T - t)))
        Bphi6 = - 0.5 * torch.pow(self.alpha1 / self.gamma, 2) * (torch.exp(-2 * self.gamma * (T - t)) - 1)

        # sum_i(Bx_i(T-t)x_i(t))
        Bx_sum = torch.sum(Bx * x, dim=0)

        # sum_i(sum_j(B_phi_{j,i}(T-t) * phi_{j,i}(t)))
        Bphi_sum = torch.sum(Bphi1 * phi1 + Bphi2 * phi2 + Bphi3 * phi3 + Bphi4 * phi4 + Bphi5 * phi5 + Bphi6 * phi6, dim=0)

        return PT / Pt * torch.exp(Bx_sum + Bphi_sum)

    def calc_instant_fwd(self, X, t, T):
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i.reshape(1, -1) for i in X]


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

        f0 = self.r0

        return f0 + Bx_sum + Bphi_sum

    def calc_zcb_price(self, X, t, T):
        if T.dim() == 0:
            T = T.unsqueeze(0)
        T = T.reshape(-1, 1)  # T = torch.linspace(0.0, Tn, int(Tn / 0.25) + 1)

        f0t = self.calc_instant_fwd(X, t, t)
        f0T = self.calc_instant_fwd(X, t, T)
        return torch.exp(-0.5 * (f0t + f0T) * (T-t))

    def calc_short_rate(self, X, t):
        return self.calc_instant_fwd(X, t, t)

    def calc_fwd(self, X, t, delta):
        zcb_t = self.calc_zcb_price(X, t, t)
        zcb_tdt = self.calc_zcb_price(X, t, t+delta)
        return forward(zcb_t, zcb_tdt, delta)

    def calc_swap(self, X, t, delta, K=None, N=torch.tensor(1.0)):
        """t = T_0, ..., T_n (future dates)"""
        zcb = self.calc_zcb_price(X=X, t=0, T=t)
        return swap(zcb, delta, K, N)

    def calc_swap_rate(self, X, t, delta):
        """t = T_0, ..., T_n (future dates)"""
        zcb = self.calc_zcb_price(X=X, t=0, T=t)
        return swap_rate(zcb, delta)

    def Bx(self, t, T):
        Bx = self.alpha1 / self.gamma * (
                    (1 / self.gamma + self.alpha0 / self.alpha1) * (torch.exp(-self.gamma * (T - t)) - 1) + \
                    (T - t) * torch.exp(-self.gamma * (T - t)))
        return Bx

    def calc_characteristic_func(self, u, t, T0, T1, discSteps=100):
        """
            Computes the transform as given by eq. (30) Trolle-Schwartz.
            Solves the system a system of ODEs by the stoch. vars. M, N using Euler's method.
        """

        M = torch.zeros(1)
        N = torch.zeros((self.simDim, 1))
        if u.is_complex():
            M = torch.zeros(1, dtype=torch.complex64)
            N = torch.zeros((self.simDim, 1), dtype=torch.complex64)

        dTau = torch.linspace(t, T0, discSteps)

        for i in range(discSteps):
            dN = N * (-self.kappa + self.sigma * self.rho * (u * self.Bx(t, T1 - T0) + (1-u) * self.Bx(t,T0))) + \
            + 0.5 * N**2 * self.sigma**2 + 0.5 * (u**2-u) * self.Bx(t, T1 - T0)**2 + 0.5 * ((1-u)**2 - (1-u))* self.Bx(t,T0)**2 +\
                + u*(1-u) * self.Bx(t, T1 - T0) * self.Bx(t,T0)
            dN *= dTau[i]
            dM = torch.sum(N * self.kappa * self.theta) * dTau[i]

            N += dN
            M += dM

        zcb0 = self.calc_zcb_price([i[:,0,:] for i in self.x], t, T0) #todo edit time index
        zcb1 = self.calc_zcb_price([i[:,0,:] for i in self.x], t, T1)
        term1 = M
        term2 = torch.sum(N * self.x[1], dim=1)
        term3 = u * torch.log(zcb1) + (1-u)*torch.log(zcb0)

        return torch.exp(term1 + term2 + term3)

    def Gfunc(self, a, b, t, T0, T1, y):
        """
            Computes the Fourier inversion of the transform.
        """
        a = torch.tensor(a)
        b = torch.tensor(b)
        def integral(MyInf=1.0):
            def integrand(u):
                c = torch.complex(real=a, imag=u * b)

                term1 = self.calc_characteristic_func(c,t,T0,T1)
                term2 = torch.exp(-torch.complex(real=torch.tensor(0.0), imag=u*y))

                integrand = torch.imag( term1 * term2 )
                return integrand / u

            du = torch.linspace(0.1, MyInf, steps=100)
            lst = [integrand(u) for u in du]
            integrands = torch.tensor(lst)

            return torch.trapz(integrands)

        term1 = 0.5 * self.calc_characteristic_func(a,t, T0, T1)
        term2 = 1/torch.pi * integral()

        return term1 - term2

    def calc_cpl(self, t, T0, T1, K):
        """
           Revise Trolle-Schwartz on this
                Cpl(0; t, T0, T1, K) = K * G_{0,1} * log(K) - G_{1,1} * log(K)
        """

        term1 = K * self.Gfunc(0.0, 1.0, t, T0, T1, torch.log(K))
        term2 = self.Gfunc(1.0, 1.0, t, T0, T1, torch.log(K))

        return term1 - term2

    def calc_cap(self, x, t, delta, K):
        """Cp(0, t, t+delta) = sum_{i=1}^n Cpl(t; Ti_1, Ti) """
        return torch.sum(self.calc_cpl(x, t, delta, K))