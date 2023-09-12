import torch
from application.engine.mcBase import Product, SampleDef


class Caplet(Product):
    def __init__(self,
                 strike: torch.Tensor,
                 start: torch.Tensor,
                 expiry: torch.Tensor,
                 delta: torch.Tensor):
        self.strike = strike
        self.start = start
        self.expiry = expiry
        self.delta = delta

        self._timeline = expiry
        self._defline = SampleDef(
            fwdMats=start,
            discMats=expiry
        )
        self._payoffLabels = '1'

    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, fwd):
        return self.delta * torch.maximum(fwd - self.strike, torch.tensor(0.0))


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

        self._timeline = torch.linspace(float(start), float(expiry), int((expiry-start)/delta+1))
        self._defline = SampleDef(
            fwdMats=self.timeline[:-1],
            discMats=self.timeline[1:]
        )
        self._payoffLabels = [str(float(t)) + 'y' for t in self.timeline[:-1]]

    @property
    def timeline(self):
        return self._timeline

    @property
    def defline(self):
        return self._defline

    @property
    def payoffLabels(self):
        return self._payoffLabels

    def payoff(self, fwd):
        return torch.sum()




if __name__ == '__main__':
    cpl = Caplet(
        strike=torch.tensor(0.05),
        expiry=torch.tensor(1.0),
        delta=torch.tensor(0.25)
    )

    print(cpl.defline)

