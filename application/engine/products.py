import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from application.utils.torch_utils import max0, smoothing
from application.utils.prd_name_conventions import float_to_time_str, float_to_notional_str
from application.engine.linearProducts import forward_rate_agreement


@dataclass
class InterestRateSwapDef:
    """ Swap(t; t+delta, N) """
    fixingDates:   torch.Tensor
    fixRate:       torch.Tensor
    notional:      torch.Tensor = torch.tensor(1.0)

    def __post_init__(self):
        self.delta = (self.fixingDates[1] - self.fixingDates[0]).view(1)  # Assumes constant delta
        self.t = self.fixingDates.reshape(-1, 1)

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
    stateVar:   bool = False


@dataclass
class Sample:
    """ Collection of market observables on an event date `t` """
    fwd:        list[torch.Tensor]
    irs:        list[torch.Tensor]
    disc:       list[torch.Tensor]
    numeraire:  torch.Tensor
    x:          torch.Tensor


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
    def name(self):
        pass

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


class CallableProduct(Product):
    @abstractmethod
    def exercise_value(self, paths: Scenario):
        pass

    @abstractmethod
    def set_exercise_idx(self, exercise_idx: torch.Tensor):
        pass

    @property
    @abstractmethod
    def exercise_dates(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def exercise_idx(self) -> torch.Tensor:
        pass


class CapletAsPutOnZCB(Product):
    def __init__(self,
                 strike: torch.Tensor,
                 exerciseDate: torch.Tensor,
                 delta: torch.Tensor,
                 notional: torch.Tensor):
        """
            This is essentially the same as a caplet, just modeled differently.
        """
        self.strike = strike
        self.exerciseDate = exerciseDate
        self.delta = delta
        self.notional = notional

        self._Tn = exerciseDate
        self._name = 'Caplet As Put On ZCB'
        self._timeline = torch.concat([torch.tensor([0.0]), exerciseDate.view(1)])
        self._defline = [
            SampleDef(
                fwdRates=[],
                irs=[],
                discMats=torch.tensor([]),
                numeraire=True
            ),
            SampleDef(
                fwdRates=[],
                irs=[],
                discMats=torch.tensor([exerciseDate + delta]),
                numeraire=True
            )
        ]
        self._payoffLabels = []

    @property
    def Tn(self):
        return self._Tn

    @property
    def name(self):
        return self._name

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
        K_bar = 1.0 + self.delta * self.strike
        payment = K_bar * max0(1.0 / K_bar - paths[1].disc[0])
        return self.notional * payment * paths[1].disc[0] * paths[0].numeraire / paths[1].numeraire


class EuropeanShortRateCall(Product):
    def __init__(self,
                 strike: torch.Tensor,
                 exerciseDate: torch.Tensor,
                 notional: torch.Tensor = torch.tensor(1.0)):
        """
            A caplet pays
                N * delta * max{ F(t, t+delta) - K; 0.0 }   @   t + delta
        """
        self.strike = strike
        self.exerciseDate = exerciseDate
        self.notional = notional
        self._Tn = exerciseDate
        self._name = f'{float_to_time_str(exerciseDate)} EuSRcall @ {strike} w {float_to_notional_str(notional)}'

        # Discounting the payment from T+delta to T
        self._timeline = torch.concat([torch.tensor([0.0]), exerciseDate.view(1)])
        self._defline = [
            SampleDef(
                fwdRates=[],
                irs=[],
                discMats=torch.tensor([]),
                numeraire=True
            ),
            SampleDef(
                fwdRates=[],
                irs=[],
                discMats=torch.tensor([]),
                numeraire=True,
                stateVar=True
            )
        ]

        self._payoffLabels = ''

    @property
    def Tn(self):
        return self._Tn

    @property
    def name(self):
        return self._name

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
        return self.notional * max0(paths[1].x - self.strike) * paths[0].numeraire / paths[1].numeraire


class Caplet(Product):
    def __init__(self,
                 strike: torch.Tensor,
                 start: torch.Tensor,
                 delta: torch.Tensor,
                 notional: torch.Tensor = torch.tensor(1.0)):
        """
            A caplet pays
                N * delta * max{ F(t, t+delta) - K; 0.0 }   @   t + delta
        """
        self.strike = strike
        self.start = start
        self.delta = delta
        self.notional = notional
        self._Tn = start + delta
        self._name = (f'{float_to_time_str(start)}'
                      f'{float_to_time_str(delta)} '
                      f'Caplet @ {float(strike) * 100:.4g}%'
                      f'w {float_to_notional_str(notional)}')

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

        self._payoffLabels = [f'{delta}*max[F({start},{start + delta})-{strike} ; 0.0]']

    @property
    def Tn(self):
        return self._Tn

    @property
    def name(self):
        return self._name

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
        # return self.notional * self.delta * max0(paths[1].fwd[0] - self.strike) * paths[0].numeraire / paths[2].numeraire
        return self.notional * self.delta * max0(paths[1].fwd[0] - self.strike) * paths[0].numeraire / paths[1].numeraire * paths[1].disc[0]


class Cap(Product):
    def __init__(self,
                 strike:            torch.Tensor,
                 firstFixingDate:   torch.Tensor,
                 lastFixingDate:    torch.Tensor,
                 delta:             torch.Tensor,
                 notional:          torch.Tensor = torch.tensor(1.0)):
        self.strike = strike
        self.firstFixingDate = firstFixingDate
        self.lastFixingDate = lastFixingDate
        self.delta = delta
        self.notional = notional

        self._Tn = lastFixingDate + delta
        self._name = (f'{float_to_time_str(firstFixingDate)}'
                      f'{float_to_time_str(lastFixingDate)}'
                      f'{float_to_time_str(delta)} '
                      f'Cap @ {float(strike) * 100:.4g}%'
                      f'w {float_to_notional_str(notional)}')

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
    def name(self):
        return self._name

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
        res = [self.notional * self.delta * max0(paths[i].fwd[0] - self.strike) * paths[0].numeraire / paths[i+1].numeraire
               for i in range(1, len(paths)-1)]  # No cashflows for the first and last sample
        '''
        res = [self.notional * self.delta * max0(paths[i].fwd[0] - self.strike) * paths[0].numeraire / paths[i].numeraire * paths[i].disc[0]
               for i in range(1, len(paths))]
        return torch.vstack(res)


class Fraption(Product):
    def __init__(self,
                 exerciseDate: torch.Tensor,
                 strike: torch.Tensor,
                 start: torch.Tensor,
                 delta: torch.Tensor,
                 notional: torch.Tensor = torch.tensor(1.0)):
        """
            A caplet pays
                delta * max{ F(t, t+delta) - K; 0.0 }   @   t + delta
        """
        self.exerciseDate = exerciseDate
        self.strike = strike
        self.start = start
        self.delta = delta
        self.notional = notional
        self._Tn = start
        self._name = (f'{float_to_time_str(start)}'
                      f'{float_to_time_str(delta)} '
                      f'Fraption @ {float(strike) * 100:.4g}%')

        self._timeline = torch.concat([torch.tensor([0.0]), exerciseDate.view(1)])
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
                discMats=torch.tensor([start]),
                numeraire=True
            )
        ]

        self._payoffLabels = [f'max[FRA({exerciseDate},{start},{start + delta}, {strike}) ; 0.0]']

    @property
    def Tn(self):
        return self._Tn

    @property
    def name(self):
        return self._name

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
        fra = forward_rate_agreement(zcb=paths[1].disc[0],
                                     fwd=paths[1].fwd[0],
                                     delta=self.delta,
                                     K=self.strike,
                                     N=self.notional)
        return max0(fra) * paths[0].numeraire / paths[1].numeraire


class EuropeanPayerSwaption(Product):
    def __init__(self,
                 strike:                torch.Tensor,
                 exerciseDate:          torch.Tensor,
                 delta:                 torch.Tensor,
                 swapFirstFixingDate:   torch.Tensor,
                 swapLastFixingDate:    torch.Tensor,
                 notional:              torch.Tensor = torch.tensor([1.0])):
        self.strike = strike
        self.exerciseDate = exerciseDate
        self.delta = delta
        self.swapFirstFixingDate = swapFirstFixingDate
        self.swapLastFixingDate = swapLastFixingDate
        self.notional = notional

        self._name = (f'{float_to_time_str(swapFirstFixingDate)}'
                      f'{float_to_time_str(swapLastFixingDate)}'
                      f'{float_to_time_str(delta)} '
                      f'EuPayerSwpt @ {float(strike) * 100:.4g}% '
                      f'w {float_to_notional_str(notional)} '
                      f'x on {float_to_time_str(exerciseDate)}')

        self._timeline = torch.concat([torch.tensor([0.0]), exerciseDate.view(1)])

        swapFixingDates = torch.linspace(
            float(self.swapFirstFixingDate),
            float(self.swapLastFixingDate),
            int((self.swapLastFixingDate - self.swapFirstFixingDate) / self.delta) + 1
        )

        self._defline = [
            SampleDef(fwdRates=[], irs=[], discMats=torch.tensor([]),
                      numeraire=True,
                      stateVar=False)
        ]

        self._defline += [
            SampleDef(
                fwdRates=[],
                irs=[InterestRateSwapDef(fixingDates=swapFixingDates, fixRate=self.strike, notional=self.notional)],
                discMats=torch.tensor([]),
                numeraire=True,
                stateVar=False
            )
        ]

        self._payoffLabels = [f'max[swap({exerciseDate}) ; 0.0]']

    @property
    def timeline(self):
        return self._timeline

    @property
    def name(self):
        return self._name

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        res = max0(paths[1].irs[0]) * paths[0].numeraire / paths[1].numeraire
        return res


class EuropeanReceiverSwaption(Product):
    def __init__(self,
                 strike:                torch.Tensor,
                 exerciseDate:          torch.Tensor,
                 delta:                 torch.Tensor,
                 swapFirstFixingDate: torch.Tensor,
                 swapLastFixingDate:    torch.Tensor,
                 notional:              torch.Tensor = torch.tensor([1.0])):
        self.strike = strike
        self.exerciseDate = exerciseDate
        self.delta = delta
        self.swapFirstFixingDate = swapFirstFixingDate
        self.swapLastFixingDate = swapLastFixingDate
        self.notional = notional

        self._timeline = torch.concat([torch.tensor([0.0]), exerciseDate.view(1)])

        self._name = (f'{float_to_time_str(swapFirstFixingDate)}'
                      f'{float_to_time_str(swapLastFixingDate)}'
                      f'{float_to_time_str(delta)} '
                      f'EuReceiverSwpt @ {float(strike) * 100:.4g}% '
                      f'w {float_to_notional_str(notional)} '
                      f'x on {float_to_time_str(exerciseDate)}')

        swapFixingDates = torch.linspace(
            float(self.swapFirstFixingDate),
            float(self.swapLastFixingDate),
            int((self.swapLastFixingDate - self.swapFirstFixingDate) / self.delta) + 1
        )

        self._defline = [
            SampleDef(fwdRates=[], irs=[], discMats=torch.tensor([]),
                      numeraire=True,
                      stateVar=False)
        ]

        self._defline += [
            SampleDef(
                fwdRates=[],
                irs=[InterestRateSwapDef(fixingDates=swapFixingDates, fixRate=self.strike, notional=self.notional)],
                discMats=torch.tensor([]),
                numeraire=True,
                stateVar=False
            )
        ]
        self._payoffLabels = [f'max[swap({exerciseDate}) ; 0.0]']

    @property
    def timeline(self):
        return self._timeline

    @property
    def name(self):
        return self._name

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        res = max0(-paths[1].irs[0]) * paths[0].numeraire / paths[1].numeraire
        return res


class BasketEuropeanPayerSwaptions(Product):
    def __init__(self,
                 exerciseDate:          torch.Tensor,
                 fixRates:              torch.Tensor,
                 deltas:                torch.Tensor,
                 swapFirstFixingDates:  torch.Tensor,
                 swapLastFixingDates:   torch.Tensor,
                 weights:               torch.Tensor,
                 notionals:             torch.Tensor,
                 strike:                torch.Tensor = torch.tensor([0.0])):

        if not (len(fixRates) == len(deltas) == len(swapFirstFixingDates) == len(swapLastFixingDates) == len(weights) == len(notionals)):
            raise ValueError(f'Length must be equal for: '
                             f'fixRates ({len(fixRates)}), '
                             f'deltas ({len(deltas)}), '
                             f'swapFirstFixingDates ({len(swapFirstFixingDates)}), '
                             f'swapLastFixingDates ({len(swapLastFixingDates)}), '
                             f'weights ({len(weights)}),'
                             f'notionals ({len(notionals)}')

        self.n = len(weights)
        self.strike = strike
        self.notionals = notionals
        self.exerciseDate = exerciseDate

        self.fixRates = fixRates
        self.deltas = deltas
        self.swapFirstFixingDates = swapFirstFixingDates
        self.swapLastFixingDates = swapLastFixingDates
        self.weights = weights

        self._name = f'Basket of European Payer Swaptions with {self.n} underlying swaps'

        self._timeline = torch.concat([torch.tensor([0.0]), self.exerciseDate.view(1)])

        self.swapFixingDates = [torch.linspace(
            float(exerciseDate),
            float(swapLastFixingDates[i]),
            int((swapLastFixingDates[i] - exerciseDate) / deltas[i] + 1)
        ) for i in range(self.n)]

        self._defline = [
            SampleDef(fwdRates=[], irs=[], discMats=torch.tensor([]),
                      numeraire=True,
                      stateVar=False)
        ]

        self._defline += [
            SampleDef(
                fwdRates=[],
                irs=[InterestRateSwapDef(fixingDates=self.swapFixingDates[i], fixRate=self.fixRates[i], notional=notionals[i])
                     for i in range(self.n)],
                discMats=torch.tensor([]),
                numeraire=True,
                stateVar=False
            )
        ]

        self._payoffLabels = ['']

    @property
    def timeline(self):
        return self._timeline

    @property
    def name(self):
        return self._name

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        irs = torch.vstack(paths[1].irs).T
        u = irs @ self.weights.reshape(-1, 1)
        res = max0(u.flatten() - self.strike) * paths[0].numeraire / paths[1].numeraire
        return res


class BermudanPayerSwaption(CallableProduct):
    def __init__(self,
                 strike:                torch.Tensor,
                 exerciseDates:         torch.Tensor,
                 delta:                 torch.Tensor,
                 swapLastFixingDate:    torch.Tensor,
                 notional:              torch.Tensor = torch.tensor([1.0])):
        self.strike = strike
        self._exerciseDates = exerciseDates
        self.delta = delta
        self.swapLastFixingDate = swapLastFixingDate
        self.notional = notional
        self._exercise_idx = None

        self._name = (f'{float_to_time_str(swapLastFixingDate)}'
                      f'{float_to_time_str(delta)} '
                      f'BermPayerSwpt @ {float(strike) * 100:.4f}% '
                      f'\w {float_to_notional_str(notional)} '
                      f'x on {[float_to_time_str(d) for d in exerciseDates]}')

        self._exerciseAtTimeZero = 0.0 in exerciseDates
        self._k = int(not self._exerciseAtTimeZero)  # Auxiliary index used in the methods `payoff` and `early_exercise`

        swapFixingDates = [torch.linspace(
            t,
            float(self.swapLastFixingDate),
            int((self.swapLastFixingDate - t) / self.delta) + 1
        ) for t in exerciseDates]

        if self._exerciseAtTimeZero:
            # First exercise is at time 0.0
            self._timeline = exerciseDates
            self._defline = [
                SampleDef(
                    fwdRates=[],
                    irs=[InterestRateSwapDef(fixingDates=swapFixingDates[j], fixRate=self.strike,
                                             notional=self.notional)],
                    discMats=torch.tensor([]),
                    numeraire=True,
                    stateVar=True
                ) for j in range(len(self._timeline))
            ]
        else:
            # First exercise is after time 0.0
            self._timeline = torch.concat([torch.tensor([0.0]), exerciseDates])
            self._defline = [
                SampleDef(
                    fwdRates=[],
                    irs=[],
                    discMats=torch.tensor([]),
                    numeraire=True,
                    stateVar=False
                )
            ]
            self._defline += [
            SampleDef(
                fwdRates=[],
                irs=[InterestRateSwapDef(fixingDates=swapFixingDates[j], fixRate=self.strike, notional=self.notional)],
                discMats=torch.tensor([]),
                numeraire=True,
                stateVar=True
            ) for j in range(len(self._timeline) - 1)
        ]

        self._payoffLabels = [f'max[swap({t}) ; 0.0]' for t in exerciseDates]

    @property
    def timeline(self):
        return self._timeline

    @property
    def name(self):
        return self._name

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    @property
    def exercise_dates(self):
        return self._exerciseDates

    @property
    def exercise_idx(self) -> torch.Tensor:
        return self._exercise_idx

    def exercise_value(self, paths: Scenario):
        res = [max0(paths[j].irs[0]) * paths[0].numeraire / paths[j].numeraire for j in range(self._k, len(paths))]
        return torch.vstack(res)

    def set_exercise_idx(self, exercise_idx: torch.Tensor):
        self._exercise_idx = exercise_idx

    def payoff(self, paths: Scenario):
        N = len(paths[0].numeraire)

        irs = torch.vstack([s.irs[0] for s in paths[self._k:]])
        df = torch.vstack([paths[0].numeraire / s.numeraire for s in paths[self._k:]])

        M = len(paths[self._k:])
        mask = torch.full(size=(M, N), fill_value=False)
        mask[self.exercise_idx.to(torch.int32), torch.arange(N, dtype=torch.int)] = True
        res = max0(irs * mask) * df
        return res


class BarrierPayerSwaption(Product):
    def __init__(self,
                 strike:                torch.Tensor,
                 exerciseDate:          torch.Tensor,
                 delta:                 torch.Tensor,
                 barrier:               torch.Tensor,
                 obsTL:                 torch.Tensor,
                 swapFirstFixingDate:   torch.Tensor,
                 swapLastFixingDate:    torch.Tensor,
                 notional:              torch.Tensor = torch.tensor([1.0]),
                 smooth:                torch.Tensor = torch.tensor([0.01]),
                 smoothing:             str = None):
        self.strike = strike
        self.exerciseDate = exerciseDate
        self.delta = delta
        self.barrier = barrier
        self.obsTL = obsTL
        self.swapFirstFixingDate = swapFirstFixingDate
        self.swapLastFixingDate = swapLastFixingDate
        self.notional = notional
        self.smooth = smooth
        self.smoothing = smoothing

        self._name = (f'{float_to_time_str(swapFirstFixingDate)}'
                      f'{float_to_time_str(swapLastFixingDate)}'
                      f'{float_to_time_str(delta)} '
                      f'BarrierPayerSwpt @ {float(strike) * 100:.4g}% '
                      f'w barrier={barrier} '
                      f'smoothing={smooth * 100}% '
                      f'and {float_to_notional_str(notional)} '
                      f'x on {float_to_time_str(exerciseDate)}')

        self._timeline = torch.unique(torch.concat([torch.tensor([0.0]), obsTL, exerciseDate.view(1)]))

        swapFixingDates = torch.linspace(
            float(self.swapFirstFixingDate),
            float(self.swapLastFixingDate),
            int((self.swapLastFixingDate - self.swapFirstFixingDate) / self.delta) + 1
        )

        self._defline = [
            SampleDef(fwdRates=[], irs=[], discMats=torch.tensor([]), numeraire=True, stateVar=False)
        ]
        self._defline += [
            SampleDef(fwdRates=[],
                      irs=[InterestRateSwapDef(fixingDates=swapFixingDates, fixRate=self.strike, notional=self.notional)],
                      discMats=torch.tensor([]),
                      numeraire=(t == self.timeline[-1]),
                      stateVar=False)
            for t in self.timeline[1:]
        ]

        self._payoffLabels = [f'max[swap({exerciseDate}) ; 0.0]']

    @property
    def timeline(self):
        return self._timeline

    @property
    def name(self):
        return self._name

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        lb = self.barrier * (1.0 - self.smooth)
        ub = self.barrier * (1.0 + self.smooth)
        alive = torch.ones_like(paths[1].irs[0])
        for sample in paths[1:]:
            swaps = sample.irs[0]
            alive *= torch.where((swaps > lb), smoothing(type=self.smoothing, x=swaps, barrier=self.barrier, lb=lb, ub=ub), alive)
            alive = torch.where(swaps > ub, 0.0, alive)


        res = max0(paths[-1].irs[0]) * paths[0].numeraire / paths[-1].numeraire
        res *= alive
        return res


class PortfolioEuropeanSwaption(Product):
    def __init__(self,
                 exerciseDates:          torch.Tensor,
                 fixRates:               list[torch.Tensor],
                 deltas:                 list[torch.Tensor],
                 swapFirstFixingDates:   list[torch.Tensor],
                 swapLastFixingDates:    list[torch.Tensor],
                 notionals:              list[torch.Tensor],
                 weights:                list[torch.Tensor],
                 flag_payer_receiver:    list[torch.Tensor]):
        """
        Portfolio of European Payer Swaptions.
        We consider
            - a vector of exercise dates and
            - a vector of vectors for all other parameters.
        """

        if not (len(fixRates) == len(exerciseDates) == len(deltas) == len(swapFirstFixingDates) == len(swapLastFixingDates) == len(notionals) == len(weights) == len(flag_payer_receiver)):
            raise ValueError(f'All arguments must have the same lengths, got:'
                             f'exerciseDates ({len(exerciseDates)}),'
                             f'strikes ({len(fixRates)}),'
                             f'deltas ({len(deltas)}),'
                             f'swapFirstFixingDates ({len(swapFirstFixingDates)}),'
                             f'swapLastFixingDates ({len(swapLastFixingDates)}),'
                             f'weights (len{weights}')

        self.numT = len(exerciseDates)
        self.numTrades = sum(len(fixRates[i]) for i in range(self.numT))

        self.fixRates = fixRates
        self.exerciseDates = exerciseDates
        self.deltas = deltas
        self.swapFirstFixingDates = swapFirstFixingDates
        self.swapLastFixingDates = swapLastFixingDates
        self.notionals = notionals
        self.weights = weights
        self.flag_payer_receiver = flag_payer_receiver

        self._name = f'Portfolio of European Payer Swaptions with {self.numT} trades'

        self._timeline = torch.unique(
            torch.concat([torch.tensor([0.0]), self.exerciseDates]), sorted=True
        )

        self.swapFixingDates = [[torch.linspace(
            float(self.swapFirstFixingDates[i][j]),
            float(self.swapLastFixingDates[i][j]),
            int((self.swapLastFixingDates[i][j] - self.swapFirstFixingDates[i][j]) / self.deltas[i][j]) + 1
        ) for j in range(len(fixRates[i]))] for i in range(self.numT)]

        self._defline = [
            SampleDef(fwdRates=[], irs=[], discMats=torch.tensor([]),
                      numeraire=True,
                      stateVar=False)
        ]

        self._defline += [
            SampleDef(
                fwdRates=[],
                irs=[InterestRateSwapDef(fixingDates=self.swapFixingDates[i][j],
                                         fixRate=self.fixRates[i][j],
                                         notional=self.notionals[i][j])
                     for j in range(len(fixRates[i]))],
                discMats=torch.tensor([]),
                numeraire=True,
                stateVar=False
            ) for i in range(self.numT)]

        self._payoffLabels = [
            f'{weights[i][j]} * max[{flag_payer_receiver[i][j]} * swap({float_to_time_str(exerciseDates[i])}, '
            f'{float_to_time_str(swapFirstFixingDates[i][j])}, '
            f'{float_to_time_str(swapLastFixingDates[i][j])}, '
            f'{float_to_time_str(deltas[i][j])})) ; 0.0]'
            for i in range(self.numT) for j in range(len(fixRates[i]))]

    @property
    def timeline(self):
        return self._timeline

    @property
    def name(self):
        return self._name

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, paths):
        res = []
        for i in range(self.numT):
            res.append(
                [self.weights[i][j] * max0(self.flag_payer_receiver[i][j] * paths[i+1].irs[j]) * paths[0].numeraire / paths[i+1].numeraire
                for j in range(len(self.fixRates[i]))]
            )

        res = torch.vstack([item for sublist in res for item in sublist])

        return res
