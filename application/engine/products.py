import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod


def max0(x):
    return torch.maximum(x, torch.tensor(0.0))


def annuity():
    raise NotImplementedError


def forward(zcb1, zcb2, delta):
    """F(t,T,T+delta) = 1 / delta * ( P(t,T) / P(t,T+delta) - 1 )"""
    return 1 / delta * (zcb1 / zcb2 - 1)


def swap(zcb:       torch.Tensor,
         delta:     torch.Tensor,
         K:         torch.Tensor or None = None,
         N:         torch.Tensor = torch.tensor(1.0)):
    """S(0) = N * [ P(0,T0) - P(0,Tn) - K * delta * sum_{i=1}^n P(0,Ti) ]"""
    if K is None:
        K = swap_rate(zcb, delta)
    return N * (zcb[0] - zcb[-1] - K * delta * torch.sum(zcb[1:]))

def swap_rate(zcb: torch.Tensor, delta: torch.Tensor):
    """R(0) = [ P(0,T0) - P(0,Tn) ] / [delta * sum_{i=1}^n P(0,Ti) ]"""
    return (zcb[0] - zcb[-1]) / (delta * torch.sum(zcb[1:]))


@dataclass
class InterestRateSwapDef:
    """ Swap(t; t+delta, N) """
    fixingDates:   torch.Tensor
    fixRate:       torch.Tensor
    notional:      torch.Tensor = torch.tensor(1.0)


@dataclass
class ForwardRateDef:
    """ F(t; T, S) """
    startDate:  torch.Tensor
    endDate:    torch.Tensor

    def __post_init__(self):
        self.delta = self.endDate - self.startDate


@dataclass
class SampleDef:
    """ Definition of what must be sampled on an event date `t` """
    fwdRates:   list[ForwardRateDef]
    irs:        list[InterestRateSwapDef]


@dataclass
class Sample:
    """ Collection of market observables on an event date `t` """
    fwd:        list[torch.Tensor]
    irs:        list[torch.Tensor]


Scenario = list[Sample]


class Product(ABC):

    @property
    @abstractmethod
    def timeline(self):
        pass

    @property
    @abstractmethod
    def defline(self) -> list[SampleDef]:
        pass

    @property
    @abstractmethod
    def paymentDates(self):
        pass

    @property
    @abstractmethod
    def payoffLabels(self):
        """Labeling of the payoffs for a product"""
        pass

    @abstractmethod
    def payoff(self, paths: Scenario) -> torch.Tensor:
        pass


class Caplet(Product):
    def __init__(self,
                 strike: torch.Tensor,
                 start: torch.Tensor,
                 delta: torch.Tensor):
        self.strike = strike
        self.start = start
        self.delta = delta

        self._timeline = start
        self._defline = [SampleDef(
            fwdRates=[ForwardRateDef(start, start + delta)],
            irs=[]
        )]
        self._payoffLabels = '1'
        self._paymentDates = torch.tensor([start + delta])

    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def paymentDates(self):
        return self._paymentDates

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths: Scenario):
        res = [self.delta * max0(s.fwd[0].reshape(-1, 1) - self.strike) for s in paths]
        return torch.concat(res, dim=1)


class Cap(Product):
    def __init__(self,
                 strike: torch.Tensor,
                 start: torch.Tensor,
                 expiry: torch.Tensor,
                 delta: torch.Tensor):
        self.strike = strike
        self.start = start
        self.expiry = expiry
        self.delta = delta

        self._timeline = torch.linspace(float(start), float(expiry-delta), int((expiry-delta-start)/delta+1))
        self._defline = [SampleDef(
            fwdRates=[ForwardRateDef(t, t + delta)],
            irs=[]
        ) for t in self.timeline]
        self._payoffLabels = [f'({t}, {t+delta})' for t in self.timeline]
        self._paymentDates = self.timeline + delta

    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def paymentDates(self):
        return self._paymentDates

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        res = [self.delta * max0(s.fwd[0].reshape(-1, 1) - self.strike) for s in paths]
        return torch.concat(res, dim=1)
