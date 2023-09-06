import torch
from abc import ABC, abstractmethod

"""
    PRODUCTS
"""


class Product(ABC):
    @abstractmethod
    def __init__(self, label=None, *args, **kwargs):
        self.timeline = None
        self.label = label

    @abstractmethod
    def payoff(self, *args, **kwargs):
        pass


class ProductEuropean(Product):
    def __init__(
            self,
            strike: torch.tensor,
            expiry: torch.tensor,
            label: str or None = None
    ):
        self.label = label if label is not None else f'{type(self).__name__} with strike {strike} and expiry {expiry}'
        self.strike = strike
        self.expiry = expiry
        self.timeline = expiry

    @abstractmethod
    def payoff(self, *args, **kwargs) -> torch.tensor:
        pass


class Caplet(ProductEuropean):
    def __init__(self, strike, expiry, delta, label=None):
        self.label = label if label is not None else (f'{type(self).__name__} with strike {strike}, expiry {expiry}, '
                                                      f'and accrual period {delta}')
        super().__init__(strike=strike, expiry=expiry, label=self.label)

    def payoff(self, spot):
        return self.delta * torch.maximum(spot - self.strike, torch.tensor(0.0))


class Cap(ProductEuropean):
    def __init__(self, strike, expiry, delta, label=None):
        self.label = label if label is not None else (f'{type(self).__name__} with strike {strike}, expiry {expiry}, '
                                                      f'and accrual period {delta}')
        super().__init__(strike, expiry, self.label)

    def payoff(self, spot, t):
        one = (torch.ones_like(self.expiry) * t == self.expiry)  # Indicator function if each caplet can be exercised
        return self.delta * torch.maximum(spot - self.strike, torch.tensor(0.0)) * one


"""
    PORTFOLIOS
"""

'''
class Portfolio(ABC):
    @abstractmethod
    def __init__(self, products: list[Product]):
        self.timeline, self.sort_idx = torch.tensor([prd.timeline for prd in products]).sort()
        self.labels = [prd.label for prd in products]


class PortfolioEuropeans(Portfolio):
    def __init__(self, products: list[ProductEuropean]):
        super().__init__(products)
        self.products = products
        self.size = len(products)
        self.strikes = [prd.strike for prd in products]
        self.expries = [prd.expiry for prd in products]

    def payoff(self, spots):
        if spots.size() == torch.Size([]):
            spots = torch.full(size=[self.size], fill_value=spots)
        return [self.products[i].payoff(spot) for i, spot in enumerate(spots)]
'''

if __name__ == '__main__':
    # Example of caplet

    delta = 0.25
    T = 2.0
    expiry = torch.linspace(delta, T, int(T/delta))
    strike = torch.tensor([0.2 * len(expiry)])

    cap = Cap(strike=strike, expiry=expiry, delta=delta)
