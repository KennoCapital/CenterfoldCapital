from application.engine.mcBase import mcSim, RNG
from application.engine.products import Cap
from application.engine.vasicek import Vasicek
import torch

seed = 1234

N = 1024

a = torch.tensor(0.86)
b = torch.tensor(0.08)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)

start = torch.tensor(2.0)
delta = torch.tensor(0.25)
expiry = torch.tensor(3.0)

t = torch.linspace(float(delta), float(expiry), int(expiry/delta))

model = Vasicek(a, b, sigma, r0)
swap_rate = model.calc_swap_rate(r0, t, delta)

rng = RNG(seed=seed, use_av=True)

prd = Cap(
    strike=swap_rate,
    start=start,
    expiry=expiry,
    delta=delta
)

mcSim(prd, model, rng, N)




