import torch
from application.engine.products import Portfolio
from application.engine.model import Model


class RNG:
    """Random Number Generator"""
    def __init__(self, N, seed=None, use_av=True):
        self.seed = seed
        self.N = N
        self.use_av = use_av
        if seed is None:
            self.gen = torch.Generator().manual_seed(torch.Generator().initial_seed())
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


class Sample:
    """
        A sample is a collection of market observations on an event date for the evaluation of the payoff:
            - Numeraire         (if the event date is a payment date), and a collection of
            - Forwards
            - Discount factors  (zero coupon bonds)
            - Libors            (forward rates)
        fixed on the event date
    """


class Scenario:
    """A scenario is a collection of samples"""


class SampleDef:
    """Definition of what must be sampled"""


def mcSim(
        port:   Portfolio,
        model:  Model,
        rng:    RNG,
        N:      int,
        seed:   int):
    """
    Template algorithm for running Monte Carlo simulation

    :param port:    Portfolio of products to value
    :param model:   Model to simulate from
    :param rng:     Random number generator
    :param N:       Number of paths to simulate
    :param seed:    Seed for replication
    :return:        Payoffs
    """

