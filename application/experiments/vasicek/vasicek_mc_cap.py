from application.engine.mcBase import mcSim, RNG
from application.engine.products import Cap
from application.engine.vasicek import Vasicek
import torch

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

seed = 1234

N = 1024

a = torch.tensor(0.86)
b = torch.tensor(0.09)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)

start = torch.tensor(0.25)
delta = torch.tensor(0.25)
expiry = torch.tensor(1.0)

eTL = torch.linspace(0.0, 1.0, 1001)    # Euler time steps

t = torch.linspace(float(delta), float(expiry), int(expiry/delta))

model = Vasicek(a, b, sigma, r0, False)
swap_rate = model.calc_swap_rate(r0, t, delta)


rng = RNG(seed=seed, use_av=True)

prd = Cap(
    strike=swap_rate,
    start=start,
    expiry=expiry,
    delta=delta
)

print(
    mcSim(prd, model, rng, N)
)

print(
    model.calc_cap(r0, t, delta, swap_rate)
)
