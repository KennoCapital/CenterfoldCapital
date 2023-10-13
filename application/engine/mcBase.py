import torch
from application.engine.products import Product, CallableProduct, Scenario
from application.engine.regressor import OLSRegressor, PolynomialRegressor
from abc import ABC, abstractmethod


MEASURES = ['risk_neutral', 'terminal']


class RNG:
    """Random Number Generator"""
    def __init__(self,
                 M:         int or None = None,
                 N:         int or None = None,
                 seed:      int or None = None,
                 use_av:    bool = True,
                 simDim:    int = 1):

        self.M = M
        self.N = N
        self.seed = seed
        self.use_av = use_av
        self.simDim = simDim

        if seed is None:
            self.gen = torch.Generator()
            self.gen.seed()
        else:
            self.gen = torch.Generator().manual_seed(seed)

    def _check_av_dim(self):
        if self.use_av and self.N % 2 != 0:
            raise ValueError('Number of paths (N) must be even when using antithetic variates!')

    def gaussMat(self):
        if self.use_av:
            self._check_av_dim()
            Z = torch.randn(size=(self.M, self.N // 2), generator=self.gen)
            return torch.concat([Z, -Z], dim=1)
        return torch.randn(size=(self.M, self.N), generator=self.gen)

    def gaussCube(self):
        if self.use_av:
            self._check_av_dim()
            Z = torch.randn(size=(self.simDim, self.M, self.N // 2), generator=self.gen)
            return torch.concat([Z, -Z], dim=2).squeeze()
        return torch.randn(size=(self.simDim, self.M, self.N), generator=self.gen)

    def next_G(self):
        """Returns a vector (tensor) N Gaussian distributed variables"""
        if self.use_av:
            Z = torch.randn(size=(self.N // 2, ), generator=self.gen)
            return torch.concat([Z, -Z], dim=1)
        return torch.randn(size=(self.N, ), generator=self.gen)

    def next_U(self):
        """Returns a vector (tensor) N Uniformly distributed variables"""
        if self.use_av:
            U = torch.rand(size=(self.N // 2, ), generator=self.gen)
            return torch.concat([U, 1-U], dim=1)
        return torch.rand(size=(self.N, ), generator=self.gen)


class Model(ABC):
    @property
    @abstractmethod
    def timeline(self):
        """Timeline of product"""
        pass

    @property
    @abstractmethod
    def dTimeline(self):
        """Timeline with additional timepoints for euler discretization"""
        pass

    @property
    @abstractmethod
    def defline(self):
        """Defline (SampleDef) of product"""
        pass

    @property
    @abstractmethod
    def x(self):
        """State variables X(t)"""
        pass

    @property
    @abstractmethod
    def paths(self):
        """Returns the samples of the market variables"""
        pass

    @abstractmethod
    def allocate(self,
                 prd:       Product,
                 N:         int,
                 dTimeline: torch.Tensor):
        """Method for allocating objects and performing pre-calculations"""
        pass

    @abstractmethod
    def calc_zcb(self, *args, **kwargs):
        pass

    @abstractmethod
    def simulate(self, Z: torch.Tensor):
        pass




class LSMC:
    def __init__(self,
                 reg:           OLSRegressor,
                 use_only_itm:  bool = True):
        self.reg = reg
        self.use_only_itm = use_only_itm
        self.coef = None
        self._N = None  # Number of paths
        self._M = None  # Number of regression times
        self._eps = 1E-12
        self._min_itm = 1024

    def backward(self,
                 prd:   CallableProduct,
                 paths: Scenario):
        # Determine exercise values
        ev = prd.exercise_value(paths)

        self._M = len(prd.exercise_dates) - 1

        self._eps = 1E-12

        # Perform regression over backwards recursion and store coefficients
        w = []
        for k in range(self._M - 1, -1, -1):
            itm = ev[k] > 0.0 + self._eps
            itm = itm if (self.use_only_itm and torch.sum(itm) >= self._min_itm) else torch.ones_like(ev[k], dtype=torch.bool)
            self.reg.fit(X=paths[k + 1].x[itm], y=ev[k][itm])
            w.insert(0, self.reg.coef)

        self.coef = torch.vstack(w)

    def forward(self,
                prd:    CallableProduct,
                paths:  Scenario):

        # Exercise values
        ev = prd.exercise_value(paths)
        self._N = ev.size()[1]

        alive = torch.ones(size=(self._N, ), dtype=torch.bool)
        stopping_idx = torch.full(size=(self._N, ), fill_value=self._M, dtype=torch.int)

        for k in range(self._M):
            self.reg.set_coef(coef=self.coef[k])

            # Continuation values
            cv = self.reg.predict(X=paths[k + 1].x)
            exercise = ev[k] > cv + self._eps
            alive = torch.logical_and(alive, exercise)
            stopping_idx[exercise] = k

        return stopping_idx

def mcSimPaths(prd:    Product,
               model:  Model,
               rng:    RNG,
               N:      int,
               dTL:    torch.Tensor = torch.tensor([]),
               simDim: int = 1):

    # Allocate and initialize results, model and rng
    model.allocate(prd, N, dTL)

    # Set dimensions
    rng.N = N
    rng.M = len(model.timeline) - 1
    rng.simDim = simDim

    # Draw random variables
    Z = rng.gaussCube()

    # Simulate state variables and fwd
    paths = model.simulate(Z)

    return paths


def mcSim(
        prd:    Product,
        mdl:    Model,
        rng:    RNG,
        N:      int,
        dTL:    torch.Tensor = torch.tensor([])):
    # Simulate paths
    paths = mcSimPaths(prd, mdl, rng, N, dTL)

    # Calculate payoffs
    payoff = prd.payoff(paths)

    return payoff


def lsmcPayoff(
        prd:            CallableProduct,
        preSimPaths:    Scenario,
        paths:          Scenario,
        lsmc:           LSMC):
    # Fit estimator of continuation value using paths form the pre-simulation
    lsmc.backward(prd, preSimPaths)

    # Determine exercise indices of the paths used for pricing
    exercise_idx = lsmc.forward(prd, paths)
    prd.set_exercise_idx(exercise_idx=exercise_idx)

    # Calculate payoffs
    payoff = prd.payoff(paths)

    return payoff


def lsmcDefaultSim(
        prd:            CallableProduct,
        mdl:            Model,
        rng:            RNG,
        N:              int,
        n:              int,
        lsmc:           LSMC = None,
        reg:            OLSRegressor = None,
        use_only_itm:   bool = True,
        dTL:            torch.Tensor = torch.Tensor([])):
    if reg is None:
        reg = PolynomialRegressor()
    if lsmc is None:
        lsmc = LSMC(reg=reg, use_only_itm=use_only_itm)

    preSimPaths = mcSimPaths(prd, mdl, rng, n, dTL)
    paths = mcSimPaths(prd, mdl, rng, N, dTL)

    payoff = lsmcPayoff(prd=prd, preSimPaths=preSimPaths, paths=paths, lsmc=lsmc)

    return payoff
