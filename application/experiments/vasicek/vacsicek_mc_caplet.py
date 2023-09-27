from application.engine.mcBase import mcSim, RNG
from application.engine.products import Caplet
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

start = torch.tensor(5.0)
delta = torch.tensor(15.0)

dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

model = Vasicek(a, b, sigma, r0, True, measure)
swap_rate = torch.tensor(0.084)


rng = RNG(seed=seed, use_av=True)

prd = Caplet(
    strike=swap_rate,
    start=start,
    delta=delta
)

payoff = mcSim(prd, model, rng, N, dTL)

print(
    'MC price =', torch.mean(payoff)
)

print(
    'Analytical price =', model.calc_cpl(r0, torch.tensor([start, start + delta]), delta, swap_rate)
)
