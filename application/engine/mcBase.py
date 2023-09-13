import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod


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

@dataclass
class SampleDef:
    """
        Definition of what must be sampled (on a specific event date)
            - discMats      Maturities of the discounts on this event date, i.e. `T` in P(0, T)
            - fwdFixings    Fixing date of forwards. The `T` in F(t, T, T+delta)
            - fwdDeltas     Accuracy period of each forward, `delta` in F(,t, T, T+delta)

    """
    discMats:   torch.Tensor
    fwdFixings: torch.Tensor
    fwdDeltas:  torch.Tensor



class Product(ABC):

    @property
    @abstractmethod
    def timeline(self):
        pass

    @property
    @abstractmethod
    def defline(self):
        pass

    @property
    @abstractmethod
    def payoffLabels(self):
        """Labeling of the payoffs for a product"""
        pass

    @abstractmethod
    def payoff(self, fwd):
        pass


class Model(ABC):
    @property
    @abstractmethod
    def timeline(self):
        """Timeline of product"""
        pass

    @property
    @abstractmethod
    def hedgeTimeline(self):
        """Timeline with required hedge-points"""
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
    def disc_curve(self):
        """Discount curve for payments P(0, T)"""
        pass

    @property
    @abstractmethod
    def x(self):
        """State variables X(t)"""
        pass

    @property
    @abstractmethod
    def fwd(self):
        """FWD sampled according to the defline"""
        pass

    @abstractmethod
    def allocate(self,
                 prdTimeline:   torch.Tensor,
                 prdDefline:    SampleDef,
                 N:             int,
                 hedgeTimeline: torch.Tensor,
                 eulerTimeline: torch.Tensor):
        """Allocator / setter for prdTimeline and prdDefline"""
        pass

    @abstractmethod
    def calc_zcb(self, *args, **kwargs):
        pass

    @abstractmethod
    def simulate(self, Z):
        pass


def mcSim(
        prd:    Product,
        model:  Model,
        rng:    RNG,
        N:      int,
        hTL:    torch.Tensor or None = None,
        eTL:    torch.Tensor or None = None):

    # Allocate and initialize results, model and rng
    model.allocate(prd.timeline, prd.defline, N, hTL, eTL)

    # Set dimensions
    rng.N = N
    rng.M = len(model.timeline) - 1

    # Draw random variables
    Z = rng.gaussMat()

    # Simulate state variables and fwd
    X, fwd = model.simulate(Z)

    # Calculate payoffs
    payoff = prd.payoff(fwd)
    payoff_pv = model.disc_curve * payoff
    npv = torch.sum(payoff_pv, dim=0)
    return torch.mean(npv)

