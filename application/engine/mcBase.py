import torch
from application.engine.products import Product, SampleDef
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

    # todo consider gaussMat dependencies

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
    def eulerTimeline(self):
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
                 prdTimeline:       torch.Tensor,
                 prdDefline:        list[SampleDef],
                 N:                 int,
                 eulerTimeline:     torch.Tensor):
        """Method for allocating objects and performing pre-calculations"""
        pass

    @abstractmethod
    def calc_zcb(self, *args, **kwargs):
        pass

    @abstractmethod
    def simulate(self, Z: torch.Tensor):
        pass

def mcSim(
        prd:    Product,
        model:  Model,
        rng:    RNG,
        N:      int,
        dTL:    torch.Tensor = torch.tensor([]),
        simDim:    int = 1):

    # Allocate and initialize results, model and rng
    model.allocate(prd.timeline, prd.defline, N, dTL)

    # todo: move RNG outside
    # Set dimensions
    rng.N = N
    rng.M = len(model.timeline) - 1
    rng.simDim = simDim

    # Draw random variables
    Z = rng.gaussCube()

    # Simulate state variables and fwd
    paths = model.simulate(Z)

    # Calculate payoffs
    payoff = prd.payoff(paths)

    return payoff

    # Discount to present value
    # payoff_pv = model.disc_curve * payoff

    # Sum across times
    # npv = torch.sum(payoff_pv, dim=1)

    # Monte Carlo Estimator
    # return torch.mean(npv)
