import torch
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from copy import copy


class RNG:
    """Random Number Generator"""
    def __init__(self, simDim=None, seed=None, use_av=True):
        self.seed = seed
        self.simDim = simDim
        self.use_av = use_av
        if seed is None:
            self.gen = torch.Generator()
            self.gen.seed()
        else:
            self.gen = torch.Generator().manual_seed(seed)

    def next_G(self):
        """Returns a vector (tensor) N Gaussian distributed variables"""
        if self.use_av:
            Z = torch.randn(size=(self.simDim // 2, ), generator=self.gen)
            return torch.concat([Z, -Z])
        return torch.randn(size=(self.simDim, ), generator=self.gen)

    def next_U(self):
        """Returns a vector (tensor) N Uniformly distributed variables"""
        if self.use_av:
            U = torch.rand(size=(self.simDim // 2, ), generator=self.gen)
            return torch.concat([U, 1-U])
        return torch.rand(size=(self.simDim, ), generator=self.gen)




@dataclass
class Sample:
    """
        A sample is a collection of market observations on an event date for the evaluation of the payoff:
            - Forwards
            - Discount factors  (zero coupon bonds) fixed on the event date
    """
    fwd: torch.Tensor
    disc: torch.Tensor

    #def allocate(self):
    #    self.fwd = torch.ones_like(self.fwd) * 100.0
    #    self.disc = torch.ones_like(self.disc) * 1.0


@dataclass
class SampleDef:
    """
        Definition of what must be sampled (on a specific event date)
            - fwdMats   Maturities of forwards on this event date
            - discMats  Maturities of the discounts on this event date
    """
    fwdMats: torch.Tensor
    discMats: torch.Tensor

    #def __len__(self):
    #    return len(fields(self))

"""A scenario is a collection of samples"""
Scenario = list[Sample]

tmp = Scenario([Sample(torch.tensor(0.1), torch.tensor(0.5))])

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
    def payoff(self, *args, **kwargs):
        pass


class Model(ABC):
    @property
    @abstractmethod
    def timeline(self):
        """Timeline of product"""
        pass

    @property
    @abstractmethod
    def defline(self):
        """Defline (SampleDef) of product"""
        pass

    @property
    @abstractmethod
    def simDim(self):
        pass

    @abstractmethod
    def allocate(self, prdTimeline: torch.Tensor, prdDefline: SampleDef):
        """Allocator / setter for prdTimeline and prdDefline"""
        pass

    @abstractmethod
    def simulate(self, *args, **kwargs):
        pass



def mcSim(
        prd:    Product,
        model:  Model,
        rng:    RNG,
        N:      int):

    cModel = copy(model)
    cRng = copy(rng)

    # Allocate and initialize results, model and rng
    nPay = len(prd.payoffLabels)
    results = torch.empty(size=(N, nPay))
    cModel.allocate(prd.timeline, prd.defline)
    simDim = cModel.simDim
    cRng.simDim = simDim

    # Iterate over paths
    for i in range(N):
        Z = cRng.next_G()
        cModel.simulate(Z)





if __name__ == '__main__':
    from application.engine.products import Cap
    delta = 0.25
    T = 2.0
    expiry = torch.linspace(delta, T, int(T/delta))
    strike = torch.tensor([0.2 * len(expiry)])

    cap = Cap(strike=strike, expiry=expiry, delta=delta)

    s = Sample(
        torch.tensor(0.0),
        torch.tensor([0.1, 0.2]),
    )
    print(s)





