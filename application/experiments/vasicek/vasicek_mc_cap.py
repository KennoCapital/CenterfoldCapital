from application.engine.mcBase import mcSim, RNG
from application.engine.products import Cap
from application.engine.vasicek import Vasicek
import torch

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

seed = None

N = 50000

measure = 'risk_neutral'

a = torch.tensor(0.86)
b = torch.tensor(0.09)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)

firstFixingDate = torch.tensor(0.25)
lastFixingDate = torch.tensor(0.75)
delta = torch.tensor(0.25)

strike = torch.tensor(0.084)

dTL = torch.linspace(0.0, lastFixingDate + delta, int(50 * (lastFixingDate + delta) + 1))

model = Vasicek(a, b, sigma, r0, True, False, measure)

rng = RNG(seed=seed, use_av=True)

prd = Cap(
    strike=strike,
    firstFixingDate=firstFixingDate,
    lastFixingDate=lastFixingDate,
    delta=delta
)

t_event_dates = torch.concat([prd.timeline, (lastFixingDate).view(1)])

cashflows = mcSim(prd, model, rng, N, dTL)
print('Cashflows: \n', cashflows, '\n')

payoff = torch.sum(cashflows, dim=0)
print('Payoffs:\n', payoff, '\n')

mc_price = torch.mean(payoff)
print('MC Price =', mc_price, '\n')

print('Model price =', model.calc_cap(r0, t_event_dates, delta, strike), '\n')
