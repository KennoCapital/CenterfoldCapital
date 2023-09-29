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
    def timeline(self) -> torch.Tensor:
        pass

    @property
    def Tn(self):
        """Timepoint used for terminal measure"""
        return self.timeline[-1]

    @property
    @abstractmethod
    def defline(self) -> list[SampleDef]:
        pass

    @property
    @abstractmethod
    def payoffLabels(self):
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
        self._Tn = start + delta

        '''
        # The straight forward definition
        self._timeline = torch.concat([torch.tensor([0.0]), start.view(1), (start + delta).view(1)])
        self._defline = [
            SampleDef(
                fwdRates=[], irs=[], discMats=torch.tensor([]),
                numeraire=True
            ),
            SampleDef(
                fwdRates=[ForwardRateDef(start, start+delta)],
                irs=[], discMats=torch.tensor([]),
                numeraire=False
            ),
            SampleDef(
                fwdRates=[], irs=[], discMats=torch.tensor([]),
                numeraire=True
            )
        ]
        '''

        # Discounting the payment from T+delta to T
        self._timeline = torch.concat([torch.tensor([0.0]), start.view(1)])
        self._defline = [
            SampleDef(
                fwdRates=[],
                irs=[],
                discMats=torch.tensor([]),
                numeraire=True
            ),
            SampleDef(
                fwdRates=[ForwardRateDef(start, start + delta)],
                irs=[],
                discMats=torch.tensor([start + delta]),
                numeraire=True
            )
        ]
        self._payoffLabels = [f'{delta}*max[F({t},{t + delta})-{strike} ; 0.0]' for t in self._timeline[1]]

    @property
    def Tn(self):
        return self._Tn

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
        # return self.delta * max0(paths[1].fwd[0] - self.strike) * paths[0].numeraire / paths[2].numeraire
        return self.delta * max0(paths[1].fwd[0] - self.strike) * paths[0].numeraire / paths[1].numeraire * paths[1].disc[0]


class Cap(Product):
    def __init__(self,
                 strike:            torch.Tensor,
                 firstFixingDate:   torch.Tensor,
                 lastFixingDate:    torch.Tensor,
                 delta:             torch.Tensor):
        self.strike = strike
        self.firstFixingDate = firstFixingDate
        self.lastFixingDate = lastFixingDate
        self.delta = delta

        self._Tn = lastFixingDate + delta

        '''
       # The straight forward definition

        self._timeline = torch.concat([
            torch.tensor([0.0]),
            torch.linspace(float(firstFixingDate),
                           float(lastFixingDate + delta),
                           int((lastFixingDate-firstFixingDate+delta) / delta) + 1)
            ])

        self._defline = [
            SampleDef(
                fwdRates=[], irs=[], discMats=torch.tensor([]),
                numeraire=True
            ),
            SampleDef(
                fwdRates=[ForwardRateDef(firstFixingDate, firstFixingDate + delta)],
                irs=[], discMats=torch.tensor([]),
                numeraire=False
            )
        ]

        self._defline += [
                SampleDef(
                    fwdRates=[ForwardRateDef(t, t + delta)],
                    irs=[], discMats=torch.tensor([]),
                    numeraire=True
                ) for t in self.timeline[2:-1]
        ]

        self._defline += [
            SampleDef(
                fwdRates=[], irs=[], discMats=torch.tensor([]),
                numeraire=True
            )
        ]
        self._payoffLabels = [f'{delta}*max[F({t},{t+delta})-{strike} ; 0.0]' for t in self._timeline[1:-1]]

        '''
        # Discounting the payment from T+delta to T
        self._timeline = torch.concat([
            torch.tensor([0.0]),
            torch.linspace(float(firstFixingDate),
                           float(lastFixingDate),
                           int((lastFixingDate - firstFixingDate) / delta + 1))
        ])

        self._defline = [
            SampleDef(
                fwdRates=[], irs=[], discMats=torch.tensor([]),
                numeraire=True
            )
        ]

        self._defline += [
            SampleDef(
                fwdRates=[ForwardRateDef(t, t + delta)],
                irs=[],
                discMats=torch.tensor([t + delta]),
                numeraire=True
            ) for t in self.timeline[1:]
        ]
        self._payoffLabels = [f'{delta}*max[F({t},{t + delta})-{strike} ; 0.0]' for t in self._timeline[1:]]

    @property
    def timeline(self):
        return self._timeline

    @property
    def Tn(self):
        return self._Tn

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        '''
        res = [self.delta * max0(paths[i].fwd[0] - self.strike) * paths[0].numeraire / paths[i+1].numeraire
               for i in range(1, len(paths)-1)]  # No cashflows for the first and last sample
        '''
        res = [self.delta * max0(paths[i].fwd[0] - self.strike) * paths[0].numeraire / paths[i].numeraire * paths[i].disc[0]
               for i in range(1, len(paths))]
        return torch.vstack(res)

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
