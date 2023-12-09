import torch
from application.engine.mcBase import Model, MEASURES
from application.engine.products import Product, Sample
from application.engine.linearProducts import forward, swap, swap_rate
from scipy.integrate import solve_ivp


def _format_dim_X(X):
    """Helper function ensuring that r0 is formatted as a row-vector"""
    X = torch.stack(X)
    if X.dim() > 1:
        raise ValueError(f'Expected r0 to be a scalar or 1D tensor (row-vector) got {X.shape}')
    if X.dim() == 0:
        X = X.unsqueeze(0)
    return X


def _format_dim_t(t):
    """Helper function ensuring that t is formatted as column-vector"""
    if t.dim() > 2:
        raise ValueError(f'Expected t to be a scalar or 2D tensor (column-vector) got {t.shape}')
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.dim() == 1:
        t = t.reshape(-1, 1)
    return t


class trolleSchwartz(Model):
    """
        Implementation of the model descibed in Trolle & Schwartz (2006):

        "A general stochastic volatility model for the pricing
            and forecasting of interest rate derivatives."
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=912447

        This code instance is currently capable of:
        - MC simulation of the model's state variables (under risk-neutral measure),
        - Semi-analytical pricing of zero-coupon bond options (caplets/floorlets).

        To be added in the future:
        - MC simulation under forward/terminal measure,
        - Semi-analytical pricing of coupon bond options (swaptions).
    """
    def __init__(self,
                 v0,
                 gamma,
                 kappa,
                 theta,
                 rho,
                 sigma,
                 alpha0,
                 alpha1,
                 varphi,
                 simDim:    int = 1,
                 measure:   str = 'risk_neutral',
                 disc_method: str = 'euler',
                 numSettings: dict = None):
        """
        :param gamma:           Parameter originating from HJM  volatility structure

        :param kappa:           Mean reversion rate variance process
        :param theta:           Long term mean level variance process
        :param rho:             Correlation between stochastic processes
        :param sigma:           Volatility parameter variance process,

        :param alpha0:          Parameter originating from HJM volatility structure
        :param alpha1:          Parameter originating from HJM volatility structure
        :param varphi:          Initial value of instantaneous forward rate f(0,T)

        :param simDim:          Multi-factor dimension
        :param numRV:           Number of stochastic processes

        :param measure:         *Supports currently only 'risk_neutral'* Specifies which measure to simulate under
        :param disc_method:     Discretization scheme of the evolution of the state variables.
                                Supports currently Euler (first order), Milstein (second order)
        """
        self._v0 = v0.reshape(-1, 1)
        self.gamma = gamma.reshape(-1 ,1)
        self.kappa = kappa.reshape(-1, 1)
        self.theta = theta.reshape(-1, 1)
        self.rho = rho.reshape(-1, 1)
        self.sigma = sigma.reshape(-1, 1)
        self.alpha0 = alpha0.reshape(-1, 1)
        self.alpha1 = alpha1.reshape(-1, 1)

        self._x0 = torch.zeros(size=(simDim, 1))
        self._phi1_0 = torch.clone(self._x0)
        self._phi2_0 = torch.clone(self._x0)
        self._phi3_0 = torch.clone(self._x0)
        self._phi4_0 = torch.clone(self._x0)
        self._phi5_0 = torch.clone(self._x0)
        self._phi6_0 = torch.clone(self._x0)

        self.varphi = varphi

        self.simDim = simDim
        self._numRV = simDim * 2

        self.measure = measure
        self.disc_method = disc_method

        if measure not in MEASURES:
            raise NotImplementedError(f'The measure "{measure}" is not implemented. '
                                      f'Use one of the following measures: {MEASURES}')

        if disc_method not in ['euler', 'milstein']:
            raise NotImplementedError(f'The discretization scheme "{disc_method}" is not implemented.'
                                      f'Use one of the following schemes: {["euler", "milstein"]}')

        if disc_method == 'milstein' and simDim > 1:
            raise NotImplementedError('Milstein discretization scheme is not supported for simDim > 1!')

        # Attributes for Monte Carlo
        self._timeline = None
        self._defline = None
        self._x = None
        self._paths = None
        self._dTimeline = None
        self._tl_idx_mkt = None
        self._Tn = None

        # Settings for numerical procedures
        self.numSettings = {
            'fourierLower': 1e-12,
            'fourierUpper': 8000.0,
            'fourierNumSteps': 50,
            'rungeKuttaNumStepsPerYear': 20,

        }
        if numSettings is not None:
            self.numSettings.update(numSettings)

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
    def x0(self):
        return [self._x0, self._v0, self._phi1_0, self._phi2_0, self._phi3_0, self._phi4_0, self._phi5_0, self._phi6_0]

    @property
    def x(self):
        return [self._x, self._v, self._phi1, self._phi2, self._phi3, self._phi4, self._phi5, self._phi6]

    @property
    def f(self):
        return self.calc_instant_fwd(X=self.x, t=torch.tensor(0.0), T=self.timeline)

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
                x=torch.full(size=(N, 8, self.simDim), fill_value=torch.nan) if prd.defline[j].stateVar else None
            ) for j in range(len(prd.timeline))
        ]

        # Specify indices of when to compute market variables
        self._tl_idx_mkt = [t in prd.timeline for t in self.timeline]

    def _correlatedBrownians(self, Z):
        """ Generate correlated Brownian motions
            Note: RV's Z is generated outside class
        """
        '''
        # Covariance matrix
        covMat = torch.ones((self.simDim, 2, 2))
        for i in range(self.simDim):
            covMat[i, 0, 1] = self.rho[i]
            covMat[i, 1, 0] = self.rho[i]

        # Cholesky decomposition: covMat = L x L^T
        L = torch.linalg.cholesky(covMat)

        # (Pairwise) Correlated BMs: W = Z x L^T
        W = torch.full_like(Z, torch.nan)
        for i in range(self.simDim):
            W[[2*i, 2*i+1], :, :] = (Z[[2*i, 2*i+1], :, :].T @ L[i].T).T

        '''

        W = Z.clone()
        for i in range(self.simDim):
            W[[2*i+1], :, :] = self.rho[i] * Z[[2*i], :, :] * torch.sqrt(1.0 - self.rho[i] ** 2.0) * Z[[2*i+1], :, :]

        Wf = W[0::2, :, :]
        Wv = W[1::2, :, :]

        return Wf, Wv

    def fwd_rate_vol(self, t, T):
        """sigma(0,t) = (alpha0 + alpha1(T-t)) * exp^{ -gamma *(T-t) }"""
        ret = [(self.alpha0[i] + self.alpha1[i] * (T-t)) * torch.exp(-self.gamma[i] * (T-t)) for i in range(self.simDim)]
        return torch.stack(ret)

    def _euler_step(self, x, v, phi1, phi2, phi3, phi4, phi5, phi6, dt, Wf, Wv):
        """
        Euler's discretisation of the state variables:
        x, v, phi1, phi2, phi3, phi4, phi5, phi6
        """
        v = torch.abs(v)
        dx = -self.gamma * x * dt + torch.sqrt(v) * Wf * torch.sqrt(dt)

        dv = self.kappa * (self.theta.reshape(self.simDim, 1) - v) * dt + self.sigma * torch.sqrt(v) * Wv * torch.sqrt(dt)

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

        return x, v, phi1, phi2, phi3, phi4, phi5, phi6

    def _milstein_step(self, x, v, phi1, phi2, phi3, phi4, phi5, phi6, dt, Wf, Wv):
        """
        Milstein's discretisation of the state variables:
        x, v, phi1, phi2, phi3, phi4, phi5, phi6
        """
        dx = -self.gamma @ x * dt + \
             torch.sqrt(v) * Wf * torch.sqrt(dt) + 0.5 * v * (torch.pow(Wf * torch.sqrt(dt), 2.0) - dt)

        dv = self.kappa @ (self.theta.reshape(self.simDim, 1) - v) * dt + \
             self.sigma @ torch.sqrt(v) * Wv * torch.sqrt(dt) + \
             0.5 * torch.pow(self.sigma @ torch.sqrt(v), 2) * (torch.pow(Wf * torch.sqrt(dt), 2.0) - dt)

        dphi1 = (x - self.gamma @ phi1) * dt
        dphi2 = (v - self.gamma @ phi2) * dt
        dphi3 = (v - 2 * self.gamma @ phi3) * dt
        dphi4 = (phi2 - self.gamma @ phi4) * dt
        dphi5 = (phi3 - 2 * self.gamma @ phi5) * dt
        dphi6 = (2 * phi5 - 2 * self.gamma @ phi6) * dt

        x += dx
        v += dv
        phi1 += dphi1
        phi2 += dphi2
        phi3 += dphi3
        phi4 += dphi4
        phi5 += dphi5
        phi6 += dphi6

        return x, v, phi1, phi2, phi3, phi4, phi5, phi6

    def simulate(self, Z):
        # Decide function for performing simulation of state variable
        if self.disc_method == 'milstein':
            step_func = self._milstein_step
        else:
            step_func = self._euler_step

        # Calculate size of time steps
        dt = self.timeline[1:] - self.timeline[:-1]

        # Compute correlated Brownian motions
        Wf, Wv = self._correlatedBrownians(Z)

        # Initialize state variables
        # set initial values to same for all paths
        self._x[:, 0, :] = self._x0
        self._v[:, 0, :] = self._v0.flatten() # todo: make this general / check for issues
        self._phi1[:, 0, :] = self._phi1_0
        self._phi2[:, 0, :] = self._phi2_0
        self._phi3[:, 0, :] = self._phi3_0
        self._phi4[:, 0, :] = self._phi4_0
        self._phi5[:, 0, :] = self._phi5_0
        self._phi6[:, 0, :] = self._phi6_0

        # Set auxiliary index
        idx = 0
        s = 0.0

        # Integratant in numéraire under risk neutral measure
        sum_x = torch.zeros((Z.shape[2], ))

        def _fillSample(x):
            nonlocal idx, s, sum_x
            if self.measure == 'risk_neutral':
                for j in range(len(self.paths[idx].fwd)):
                    self._paths[idx].fwd[j][:] = self.calc_fwd(X=x,
                                                               t=self.defline[idx].fwdRates[j].startDate - s,
                                                               delta=self.defline[idx].fwdRates[j].delta)

                for j in range(len(self.paths[idx].irs)):
                    self._paths[idx].irs[j][:] = self.calc_swap(X=x,
                                                                t=self.defline[idx].irs[j].t - s,
                                                                delta=self.defline[idx].irs[j].delta,
                                                                K=self.defline[idx].irs[j].fixRate,
                                                                N=self.defline[idx].irs[j].notional)

                for j in range(len(self.paths[idx].disc)):
                    self._paths[idx].disc[j][:] = self.calc_zcb(X=x,
                                                                t=torch.tensor(0.0),
                                                                T=self.defline[idx].discMats[j] - s)

                if self.paths[idx].numeraire is not None:
                    self._paths[idx].numeraire[:] = torch.exp(sum_x)

                if self.paths[idx].x is not None:
                    self._paths[idx].x[:, :, :] = torch.stack(x).reshape(-1, 8, self.simDim)

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
            self._x[:, k+1, :], self._v[:,k+1, :], self._phi1[:,k+1, :], self._phi2[:,k+1, :], \
            self._phi3[:, k+1, :], self._phi4[:,k+1, :], self._phi5[:,k+1, :], self._phi6[:,k+1, :] = \
                step_func(self._x[:, k, :], self._v[:,k, :], self._phi1[:,k, :], self._phi2[:,k, :],
                          self._phi3[:,k, :], self._phi4[:,k, :], self._phi5[:,k, :], self._phi6[:,k, :],
                          dt[k], Wf[:,k,:], Wv[:,k, :])

            # Numeraire
            if self.measure == 'risk_neutral':
                # Trapezoidal rule: B(t) = exp{ int_0^t r(s) ds } ~ exp{sum[ r(t) * dt ]}
                # Get state variables
                xk1 = [self._x[:, k+1, :], self._v[:, k+1, :], self._phi1[:, k+1, :], self._phi2[:, k+1, :], \
                        self._phi3[:, k+1, :], self._phi4[:, k+1, :], self._phi5[:, k+1, :], self._phi6[:, k+1, :]]
                xk = [self._x[:, k, :], self._v[:, k, :], self._phi1[:, k, :], self._phi2[:, k, :], \
                        self._phi3[:, k, :], self._phi4[:, k, :], self._phi5[:, k, :], self._phi6[:, k, :]]

                # Calc short rates
                rt1 = self.calc_short_rate(X=xk1, t=self.timeline[k] + dt[k])
                rt = self.calc_short_rate(X=xk, t=self.timeline[k])

                # Apply trapezoidal rule
                sum_x += 0.5 * (rt1 + rt).flatten() * dt[k]

            # Samples (market variables)
            if self._tl_idx_mkt[k + 1]:
                _fillSample(x=[
                    self._x[:, k+1, :], self._v[:, k+1, :],
                    self._phi1[:, k+1, :], self._phi2[:, k+1, :],
                    self._phi3[:, k+1, :], self._phi4[:, k+1, :],
                    self._phi5[:, k+1, :], self._phi6[:, k+1, :]]
                )
        return self.paths

    def calc_zcb(self, X, t, T):
        """
        Calculate time-t price of zero-coupon bond maturing at time T:
            P(t,T) = P(0,T)/P(0,t) * exp{sum_i(Bx_i(T-t)x_i(t)) +
                        sum_i(sum_j(B_phi_{j,i}(T-t) * phi_{j,i}(t)))}
        Given by equations (20)-(27).

        :param X:    [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
        """
        if t.dim() == 0:
            t = t.view(1)
        t = t.reshape(-1, 1)
        if T.dim() == 0:
            T = T.view(1)
        T = T.reshape(-1, 1)

        # Extract
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [x.reshape(self.simDim, -1) for x in X]

        # eq. (20) - (26)
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

        # eq. (20) by each term
        zcbT_by_zcbt = torch.exp(-self.varphi * (T - t))
        Bx_sum = Bx.T @ x
        Bphi_sum = Bphi1.T @ phi1 + Bphi2.T @ phi2 + Bphi3.T @ phi3 + Bphi4.T @ phi4 + Bphi5.T @ phi5 + Bphi6.T @ phi6

        return zcbT_by_zcbt * torch.exp(Bx_sum + Bphi_sum)

    def calc_instant_fwd(self, X, t, T):
        """
        Calculate time-t instantaneous forward rate for risk-free borrowing & lending at time T:
            f(t,T) = f(0,T) + sum_i{Bx_i(T-t)*x_i(t)} + sum_i{ sum_j{Bphi_{j,i}(T-t) * phi_{j,i}(t)} },
        given by equations (5)-(12).

        :param X:    [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
        """
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i.reshape(self.simDim, -1) for i in X]

        if t.dim() == 0:
            t = t.view(1)
        t = t.reshape(-1, 1)
        if T.dim() == 0:
            T = T.view(1)
        T = T.reshape(-1, 1)

        # eq. (6) - (12)
        Bx = (self.alpha0 + self.alpha1 * (T - t)) * torch.exp(-self.gamma * (T - t))

        Bphi1 = self.alpha1 * torch.exp(-self.gamma * (T - t))

        Bphi2 = self.alpha1 / self.gamma * (1/self.gamma + self.alpha0/self.alpha1) * \
                (self.alpha0 + self.alpha1*(T-t)) * torch.exp(-self.gamma * (T - t))

        Bphi3 = - ( self.alpha0 * self.alpha1 / self.gamma * (1/self.gamma + self.alpha0/self.alpha1) + \
                    self.alpha1/self.gamma * (self.alpha1 / self.gamma + 2 * self.alpha0)*(T-t) + \
                    torch.pow(self.alpha1,2) /self.gamma * (T-t)**2 ) * torch.exp(-2 * self.gamma * (T - t))

        Bphi4 = torch.pow(self.alpha1,2) / self.gamma * (1/self.gamma + self.alpha0/self.alpha1) * torch.exp(-self.gamma * (T - t))

        Bphi5 = - self.alpha1 / self.gamma * (self.alpha1 / self.gamma + 2*self.alpha0 + 2* self.alpha1*(T-t)) * torch.exp(-2 * self.gamma * (T - t))

        Bphi6 = - torch.pow(self.alpha1, 2) / self.gamma * torch.exp(-2 * self.gamma * (T - t))

        # eq. (5) by each term
        f0T = self.varphi
        Bx_sum = Bx.T @ x
        Bphi_sum = Bphi1.T @ phi1 + Bphi2.T @ phi2 + Bphi3.T @ phi3 + Bphi4.T @ phi4 + Bphi5.T @ phi5 + Bphi6.T @ phi6

        return f0T + Bx_sum + Bphi_sum

    def calc_zcb_price(self, X, t, T):
        """
        Calculates the ZCB price as:
            P(t,T) = exp{-int_t^T f(t,u) du }
        """
        if T.dim() == 0:
            T = T.unsqueeze(0)
        T = T.reshape(-1, 1)

        f0t = self.calc_instant_fwd(X, t, t)
        f0T = self.calc_instant_fwd(X, t, T)

        return torch.exp(-0.5 * (f0t + f0T) * (T-t))

    def calc_short_rate(self, X, t):
        return self.calc_instant_fwd(X, t, t)

    def calc_fwd(self, X, t, delta):
        zcb_t = self.calc_zcb(X, t, t)
        zcb_tdt = self.calc_zcb(X, t, t + delta)

        return forward(zcb_t, zcb_tdt, delta)

    def calc_swap(self, X, t, delta, K=None, N=torch.tensor(1.0)):
        """t = T_0, ..., T_{n-1} (fixing dates)"""
        if not (delta.dim() == 0 or (delta.dim() == 1 and max(delta.size()) == 1)):
            raise NotImplementedError(f'delta must be a scalar when calculating the swaps. Got {delta.shape}')
        if delta.dim() == 0:
            delta = delta.unsqueeze(0)

        t = _format_dim_t(t)

        zcb = self.calc_zcb(X=X, t=torch.tensor(0.0), T=t)
        zcb_tdt = self.calc_zcb(X=X, t=torch.tensor(0.0), T=t[-1] + delta)

        swaps = swap(torch.concat([zcb, zcb_tdt], dim=0), delta, K, N)

        return swaps.nan_to_num(float(swaps.nanmean()))

    def calc_swap_rate(self, X, t, delta):
        """t = T_0, ..., T_n (future dates)"""
        zcb = self.calc_zcb(X=X, t=torch.tensor(0.), T=t)
        return swap_rate(zcb, delta)


    def calc_characteristic_func(self, u, t, T0, T1):
        """
            Computes the transform as given by eq. (30) Trolle-Schwartz.
            Solves the system a system of ODEs by the stoch. vars. M, N using Runge-Kutta-4.
        """
        if not u.is_complex():
            N0 = torch.zeros((self.simDim, 1))
        else:
            N0 = torch.zeros((self.simDim, 1), dtype=torch.complex64)

        def Bx(tau):
            """ eq. (21) """
            if not torch.is_tensor(tau):
                tau = torch.tensor(tau)

            Bx = self.alpha1 / self.gamma * (
                    (1 / self.gamma + self.alpha0 / self.alpha1) * (torch.exp(-self.gamma * tau) - 1) + \
                    tau * torch.exp(-self.gamma * tau))
            return Bx

        def _compute_dN(t,N):
            dN = [
                N[i] * (-self.kappa[i] + self.sigma[i] * self.rho[i] * (u * Bx(T1-T0+t)[i] + (1-u) * Bx(t)[i])) + \
                0.5 * N[i]**2 * self.sigma[i]**2 + 0.5 * (u**2-u) * Bx(T1-T0+t)[i]**2 + 0.5 * ((1-u)**2 - (1-u))* Bx(t)[i]**2 +\
                u*(1-u) * Bx(T1-T0+t)[i] * Bx(t)[i]
                for i in range(self.simDim)
            ]
            return torch.hstack(dN)

        # Solve system of ODEs with Runge-Kutta (RK45)
        numSteps = int(self.numSettings['rungeKuttaNumStepsPerYear'] * float(T0) + 1)
        dTau = torch.linspace(float(t), float(T0), numSteps)
        sol = solve_ivp(_compute_dN, t_span=[t, T0], y0=N0.flatten(), t_eval=dTau, method='RK45')
        Nmat = torch.tensor(sol.y)

        # Solutions to ODEs
        M = torch.trapz(torch.sum(Nmat * self.kappa * self.theta, dim=0), dTau, dim=0)
        N = Nmat[:, -1].reshape(self.simDim, 1)

        zcb0 = self.calc_zcb(self.x0, t, T0)
        zcb1 = self.calc_zcb(self.x0, t, T1)

        term1 = M
        term2 = torch.sum(N * self._v0, dim=0)
        term3 = u * torch.log(zcb1) + (1.0 - u) * torch.log(zcb0)

        return torch.exp(term1 + term2 + term3)

    def Gfunc(self, a, b, t, T0, T1, y):
        """
            Computes the Gil-Pelaez formula:
            G_{a,b}(y) = phi(a,t,T0,T1) / 2 - 1 / pi int{ Im(phi(a+iub,t,T0,T1) e{-iuy} / u du}
        """
        a = torch.tensor(a)
        b = torch.tensor(b)

        def integral():
            def integrand(u):
                c = torch.complex(real=a, imag=u * b)
                term1 = self.calc_characteristic_func(c, t, T0, T1)
                term2 = torch.exp(-torch.complex(real=torch.tensor(0.0), imag=u*y))
                integrand = torch.imag(term1 * term2) / u

                return integrand

            # Integration steps
            du = torch.linspace(
                start=self.numSettings['fourierLower'],
                end=self.numSettings['fourierUpper'],
                steps=self.numSettings['fourierNumSteps']
            )

            # Function values
            fu = torch.stack([integrand(u).reshape(-1) for u in du])
            return torch.trapz(fu, du, dim=0)

        return self.calc_characteristic_func(a, t, T0, T1) / 2 - 1.0 / torch.pi * integral()

    def calc_cpl(self, t, T0, delta, K, N=torch.tensor(1.0)):
        """
            Calculate the time-t price of a caplet with maturity T0 and tenor delta,
            as European put option on ZCB, eq. (33):
                Put(t, T0, T1, X) = X * G_{0,1}(log(X)) - G_{1,1}(log(X)),
                where T1 = T0 + delta,
                X = 1/(1+delta*K)

                such that
                Cpl(t,T0,delta, K) = 1/X * Put(t, T0, T1, X)
        """

        K_bar = 1.0 / (1.0 + delta * K)
        T1 = T0 + delta
        term1 = K_bar * self.Gfunc(0.0, 1.0, t, T0, T1, torch.log(K_bar))
        term2 = self.Gfunc(1.0, 1.0, t, T0, T1, torch.log(K_bar))
        return N / K_bar * (term1 - term2)

    def calc_cap(self, x, t, delta, K):
        """Cp(0, t, t+delta) = sum_{i=1}^n Cpl(t; Ti_1, Ti) """
        return torch.sum(self.calc_cpl(x, t, delta, K))
