import torch
from abc import ABC, abstractmethod

"""
    PRODUCTS
"""


class Product(ABC):
    @abstractmethod
    def __init__(self, label=None):
        self.timeline = None
        self.label = None

    @abstractmethod
    def payoff(self):
        pass


class ProductEuropean(Product):
    def __init__(self, strike, expiry, label=None):
        self.strike = strike
        self.expiry = expiry
        self.timeline = expiry
        self.label = label if label is not None else f'{type(self).__name__} with strike {strike} and expiry {expiry}'

    @abstractmethod
    def payoff(self, spot):
        pass


class Caplet(ProductEuropean):
    def __init__(self, strike, expiry, delta, label=None):
        super().__init__(strike, expiry, label)


    def payoff(self, spot):
        return torch.maximum(spot - self.strike, torch.tensor(0.0))


class Floorlet(ProductEuropean):
    def payoff(self, spot):
        return torch.maximum(self.strike - spot, torch.tensor(0.0))


"""
    PORTFOLIOS
"""


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


if __name__ == '__main__':
    # Example of caplet
    spot = torch.tensor(0.03)
    expiry = torch.tensor(1.0)
    strike = torch.tensor(0.02)

    cpl = Caplet(strike, expiry)
    print(cpl.payoff(spot))


    # Example of cap (portfolio / strip of caplets)
    cpl2 = Caplet(strike, expiry * 2)
    cap = PortfolioEuropeans(products=[cpl2, cpl])
    print(cap.payoff(torch.tensor([spot])))
    print(cap.timeline)
    print(cap.labels)