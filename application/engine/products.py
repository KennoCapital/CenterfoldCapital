import torch
from abc import ABC, abstractmethod
from application.engine.mcBase import Product, SampleDef


class ProductEuropean(Product):
    def __init__(self,
                 strike: torch.tensor,
                 expiry: torch.tensor,
                 label: str or None = None):
        self.label = label if label is not None else f'{type(self).__name__} with strike {strike} and expiry {expiry}'
        self.strike = strike
        self.expiry = expiry
        self.timeline = expiry

    @abstractmethod
    def payoff(self, *args, **kwargs) -> torch.tensor:
        pass


class Caplet(Product):
    def __init__(self,
                 strike: torch.tensor,
                 expiry: torch.tensor,
                 delta: torch.tensor,
                 label: str or None=None):
        self.defline = SampleDef(
            fwdMats=expiry,
            discMats=expiry,
            numeriare=True
        )
        self.delta = delta
        self.label = label if label is not None else (f'{type(self).__name__} with strike {strike}, expiry {expiry}, '
                                                      f'and accrual period {delta}')
        super().__init__(strike=strike, expiry=expiry, label=self.label)

    def payoff(self, spot):
        return self.delta * torch.maximum(spot - self.strike, torch.tensor(0.0))


if __name__ == '__main__':
    cpl = Caplet(
        strike=torch.tensor(0.05),
        expiry=torch.tensor(1.0),
        delta=torch.tensor(0.25)
    )

    print(cpl.defline)
