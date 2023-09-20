from application.engine.mcBase import mcSim, RNG
from application.engine.products import EuropeanPayerSwaption
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

exerciseDate = torch.tensor(0.25)
delta = torch.tensor(0.25)
swapLastFixingDate = torch.tensor(1.0)

t = torch.linspace(float(exerciseDate),
                   float(swapLastFixingDate + delta),
                   int((swapLastFixingDate + delta - exerciseDate)/delta + 1))

model = Vasicek(a, b, sigma, r0, False)
swap_rate = model.calc_swap_rate(r0, t, delta)


rng = RNG(seed=seed, use_av=True)

prd = EuropeanPayerSwaption(
    strike=swap_rate,
    exerciseDate=exerciseDate,
    swapLastFixingDate=swapLastFixingDate,
    delta=delta
)

print(
    mcSim(prd, model, rng, N)
)


