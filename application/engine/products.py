import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from application.utils.torch_utils import max0


@dataclass
class InterestRateSwapDef:
    """ Swap(t; t+delta, N) """
    fixingDates:   torch.Tensor
    fixRate:       torch.Tensor
    notional:      torch.Tensor = torch.tensor(1.0)

    def __post_init__(self):
        self.delta = (self.fixingDates[1] - self.fixingDates[0]).view(1)  # Assumes constant delta
        self.t = torch.concat([self.fixingDates, self.fixingDates[-1] + self.delta], dim=0).reshape(-1, 1)


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
    discMats:   torch.Tensor = torch.tensor([])
    numeraire:  bool = True


@dataclass
class Sample:
    """ Collection of market observables on an event date `t` """
    fwd:        list[torch.Tensor]
    irs:        list[torch.Tensor]
    disc:       list[torch.Tensor]
    numeraire:  torch.Tensor


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
        """
            A caplet pays
                delta * max{ F(t, t+delta) - K; 0.0 }   @   t + delta
        """
        self.strike = strike
        self.start = start
        self.delta = delta

        self._timeline = start.view(1)

        self._defline = [
            SampleDef(fwdRates=[
                ForwardRateDef(start, start + delta)],
                irs=[],
                discMats=torch.tensor([]),
                numeraire=True)
        ]
        self._payoffLabels = '1'
        # self._paymentDates = torch.tensor([start + delta])

    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths: Scenario):
        res = self.delta * max0(paths[0].fwd[0] - self.strike) / paths[0].numeraire
        return res


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
        self._defline = [
            SampleDef(
                fwdRates=[ForwardRateDef(t, t + delta)],
                irs=[],
                numeraire=True
            ) for t in self.timeline]
        self._payoffLabels = [f'({t}, {t+delta})' for t in self.timeline]
        # self._paymentDates = self.timeline + delta

    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        res = [self.delta * max0(s.fwd[0].reshape(-1, 1) - self.strike) for s in paths]
        return torch.concat(res, dim=1)


class EuropeanPayerSwaption(Product):
    def __init__(self,
                 strike:                torch.Tensor,
                 exerciseDate:          torch.Tensor,
                 delta:                 torch.Tensor,
                 swapLastFixingDate:    torch.Tensor,
                 swapFirstFixingDate:   torch.Tensor = torch.tensor([]),
                 notional:              torch.Tensor = torch.tensor([1.0])):
        self.strike = strike
        self.exerciseDate = exerciseDate
        self.swapLastFixingDate = swapLastFixingDate
        self.delta = delta
        self.swapFirstFixingDate = swapFirstFixingDate
        self.notional = notional

        if len(swapFirstFixingDate) == 0 or swapFirstFixingDate is None:
            self.swapFirstFixingDate = exerciseDate

        self._timeline = exerciseDate.view(1)
        swapFixingDates = torch.linspace(
            float(self.swapFirstFixingDate),
            float(self.swapLastFixingDate),
            int((self.swapLastFixingDate - self.swapFirstFixingDate) / self.delta) + 1
        )

        self._defline = [
            SampleDef(
                fwdRates=[],
                irs=[InterestRateSwapDef(fixingDates=swapFixingDates, fixRate=self.strike, notional=self.notional)],
                numeraire=True
            )
        ]
        self._payoffLabels = [f'{exerciseDate}']

    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        res = [max0(s.irs[0].reshape(-1, 1)) for s in paths]
        return torch.concat(res, dim=1)
