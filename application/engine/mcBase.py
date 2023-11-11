import torch
from application.engine.products import Product, CallableProduct, Scenario
from application.engine.regressor import OLSRegressor, PolynomialRegressor
from application.utils.torch_utils import max0
from abc import ABC, abstractmethod


MEASURES = ['risk_neutral', 'terminal']


class RNG:
    """Random Number Generator"""
    def __init__(self,
                 M:         int or None = None,
                 N:         int or None = None,
                 numRV:     int = 1,
                 seed:      int or None = None,
                 use_av:    bool = True
                 ):

        self.M = M
        self.N = N
        self.numRV = numRV
        self.seed = seed
        self.use_av = use_av

        if seed is None:
            self.gen = torch.Generator()
            self.gen.seed()
        else:
            self.gen = torch.Generator().manual_seed(seed)

    def _check_av_dim(self):
        if self.use_av and self.N % 2 != 0:
            raise ValueError('Number of paths (N) must be even when using antithetic variates!')

    def gaussCube(self):
        """Returns a Cube (3D-tensor) of numRV x M x N Gaussian random variables"""
        if self.use_av:
            self._check_av_dim()
            Z = torch.randn(size=(self.numRV, self.M, self.N // 2), generator=self.gen)
            return torch.concat([Z, -Z], dim=2)
        return torch.randn(size=(self.numRV, self.M, self.N), generator=self.gen)

    def gaussMat(self):
        """Returns a matrix (2D-tensor) of MxN Gaussian random variables"""
        if self.use_av:
            self._check_av_dim()
            Z = torch.randn(size=(self.M, self.N // 2), generator=self.gen)
            return torch.concat([Z, -Z], dim=1)
        return torch.randn(size=(self.M, self.N), generator=self.gen)

    def next_G(self):
        """Returns a vector (tensor) of N Gaussian random variables"""
        if self.use_av:
            Z = torch.randn(size=(self.N // 2, ), generator=self.gen)
            return torch.concat([Z, -Z], dim=1)
        return torch.randn(size=(self.N, ), generator=self.gen)

    def next_U(self):
        """Returns a vector (tensor) of N Uniformly distributed variables"""
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
    def numRV(self):
        """Number of Wiener Processes (Gaussian Random Variables) in the model"""
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
        self._min_itm = 2048

    def _formatPathsX(self, paths):
        return [p.x.reshape(p.x.shape[0], -1) if p.x is not None else None for p in paths]



    def backward(self,
                 prd:   CallableProduct,
                 paths: Scenario):

        # Auxiliary variable used to match the index between paths and exercise values,
        # depending on if time 0.0 is an exercise date
        idx_offset = -1 if (0.0 in prd.exercise_dates) else 0
        self._M = len(prd.exercise_dates) - 1
        self._eps = 1E-12

        # Determine exercise values
        ev = prd.exercise_value(paths)

        # Data preprocessing (arrange state variables column-wise)
        xList = self._formatPathsX(paths)

        # Perform regression over backwards recursion and store coefficients
        w = []
        for k in range(self._M, 0, -1):
            itm = ev[k] > 0.0 + self._eps
            itm = itm if (self.use_only_itm and torch.sum(itm) >= self._min_itm) else torch.ones_like(ev[k], dtype=torch.bool)
            #self.reg.fit(X=paths[k + idx_offset].x[itm], y=ev[k][itm])
            self.reg.fit(X=xList[k + idx_offset][itm], y=ev[k][itm])
            w.insert(0, self.reg.coef)
        self.coef = torch.vstack(w)

    def forward(self,
                prd:    CallableProduct,
                paths:  Scenario):
        # Exercise values
        ev = prd.exercise_value(paths)
        self._N = ev.size()[1]

        # Auxiliary variable used to match the index between paths and exercise values,
        # depending on if time 0.0 is an exercise date
        idx_offset = 0 if (0.0 in prd.exercise_dates) else 1

        # Data preprocessing (arrange state variables column-wise)
        xList = self._formatPathsX(paths)

        alive = torch.ones(size=(self._N, ), dtype=torch.bool)
        stopping_idx = torch.full(size=(self._N, ), fill_value=self._M, dtype=torch.int)
        for k in range(self._M):
            self.reg.set_coef(coef=self.coef[k])

            # Continuation values
            #cv = self.reg.predict(X=paths[k + idx_offset].x)
            cv = self.reg.predict(X=xList[k + idx_offset])
            exercise = ev[k] > max0(cv) + self._eps
            exercise = torch.logical_and(alive, exercise)
            stopping_idx[exercise] = k

        return stopping_idx


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
    rng.numRV = model.numRV

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
