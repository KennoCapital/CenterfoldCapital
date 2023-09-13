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
            zcbMats=expiry,
            fwdFixings=start,
            fwdDeltas=delta
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

    def payoff(self, zcb, fwd):
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
            zcbMats=self.timeline[1:],
            fwdFixings=self.timeline[:-1],
            fwdDeltas=self.delta * torch.ones_like(self.timeline[1:])
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

    def payoff(self, zcb, fwd):
        return self.delta * torch.maximum(fwd - self.strike, torch.tensor(0.0))



if __name__ == '__main__':
    cpl = Caplet(
        start=torch.tensor(0.25),
        strike=torch.tensor(0.05),
        expiry=torch.tensor(1.0),
        delta=torch.tensor(0.25)
    )

    cap = Cap(
        start=torch.tensor(0.25),
        strike=torch.tensor(0.05),
        expiry=torch.tensor(1.0),
        delta=torch.tensor(0.25)
    )

    print(cap.timeline)
    print(cap.defline)
