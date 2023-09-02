from abc import ABC, abstractmethod
from products import Portfolio
import torch



N = lambda x: torch.distributions.Normal(loc=0.0, scale=1.0).cdf(x)


class MCModel(ABC):
    @abstractmethod
    def __init__(self, portfolio: Portfolio):
        self.timeline = portfolio.timeline

    @abstractmethod
    def make_timeline(self):
        pass

    @abstractmethod
    def generatePaths(self):
        pass


class VasicekMCModel(MCModel):
    def __init__(self,
                 timeline:  torch.Tensor,
                 portfolio: Portfolio,
                 a:         float,
                 b:         float,
                 sigma:     float):

        self.timeline = timeline
        self.portfolio = portfolio
        self.a = a
        self.b = b
        self.sigma = sigma

    def generatePaths(self, gaussVec):
        pass


if __name__ == '__main__':
   pass