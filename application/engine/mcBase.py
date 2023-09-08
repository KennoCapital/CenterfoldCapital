import torch
from application.engine.model import Model
from dataclasses import dataclass
from abc import ABC, abstractmethod


class RNG:
    """Random Number Generator"""
    def __init__(self, N, seed=None, use_av=True):
        self.seed = seed
        self.N = N
        self.use_av = use_av
        if seed is None:
            self.gen = torch.Generator()
            self.gen.seed()
        else:
            self.gen = torch.Generator().manual_seed(seed)

    def next_G(self):
        """Returns a vector (tensor) N Gaussian distributed variables"""
        if self.use_av:
            Z = torch.randn(size=(self.N // 2, ), generator=self.gen)
            return torch.concat([Z, -Z])
        return torch.randn(size=(self.N, ), generator=self.gen)

    def next_U(self):
        """Returns a vector (tensor) N Uniformly distributed variables"""
        if self.use_av:
            U = torch.rand(size=(self.N // 2, ), generator=self.gen)
            return torch.concat([U, 1-U])
        return torch.rand(size=(self.N, ), generator=self.gen)


class Scenario:
    """A scenario is a collection of samples"""

@dataclass
class Sample:
    """
        A sample is a collection of market observations on an event date for the evaluation of the payoff:
            - Forwards
            - Discount factors  (zero coupon bonds) fixed on the event date
    """
    fwd: torch.tensor
    disc: torch.tensor

@dataclass
class SampleDef:
    """
        Definition of what must be sampled (on a specific event date)
            - fwdMats   Maturities of forwards on this event date
            - discMats  Maturities of the discounts on this event date
    """
    fwdMats: torch.tensor
    discMats: torch.tensor


class Product(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.timeline: torch.tensor
        self.defline: SampleDef
        self.label: str

    @abstractmethod
    def payoff(self, *args, **kwargs):
        pass



def mcSim(
        prd:    Product,
        model:  Model,
        rng:    RNG,
        N:      int):
    """
    Template algorithm for running Monte Carlo simulation

    :param port:    Portfolio of products to value
    :param model:   Model to simulate from
    :param rng:     Random number generator
    :param N:       Number of paths to simulate
    :return:        Payoffs
    """

    tl = prd.timeline

    # simulation
    for k, s in enumerate(tl):
        Z = rng.next_G()
        x = model.simulate(Z)

    return payoffs
    


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





