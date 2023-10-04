import torch
from application.engine.products import Product, CallableProduct, Scenario
from application.engine.regressor import OLSRegressor
from abc import ABC, abstractmethod


MEASURES = ['risk_neutral', 'terminal']


class RNG:
    """Random Number Generator"""
    def __init__(self,
                 M:         int or None = None,
                 N:         int or None = None,
                 seed:      int or None = None,
                 use_av:    bool = True):

        self.M = M
        self.N = N
        self.seed = seed
        self.use_av = use_av

        self.simDim = None
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

def mcSimPaths(prd:    Product,
               model:  Model,
               rng:    RNG,
               N:      int,
               dTL:    torch.Tensor = torch.tensor([])):

    # Allocate and initialize results, model and rng
    model.allocate(prd, N, dTL)

    # Set dimensions
    rng.N = N
    rng.M = len(model.timeline) - 1

    # Draw random variables
    Z = rng.gaussMat()

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




class LSMC:
    def __init__(self,
                 reg:   OLSRegressor):
        self.reg = reg
        self.coef = None
        self._N = None  # Number of paths
        self._M = None  # Number of regression times

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
            self.reg.fit(X=paths[k + 1].x, y=ev[k])
            w.insert(0, self.reg.coef)

        self.coef = torch.vstack(w)

    def forward(self,
                prd:    CallableProduct,
                paths:  Scenario):

        # Exercise values
        ev = prd.exercise_value(paths)
        self._N = ev.size()[1]

        alive = torch.ones(size=(self._N, ), dtype=torch.bool)
        stopping_idx = torch.full(size=(self._N, ), fill_value=torch.nan)

        for k in range(self._M):
            print(k)
            self.reg.set_coef(coef=self.coef[k])

            # Continuation values
            print('x: ', paths[k+1].x)
            cv = self.reg.predict(X=paths[k + 1].x)

            print('cv: ', cv)
            exercise = ev[k] > cv + self._eps
            alive = torch.logical_and(alive, exercise)
            stopping_idx[exercise] = k

        return stopping_idx





