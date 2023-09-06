import torch
import matplotlib.pyplot as plt
from application.engine.vasicek import Vasicek
from application.engine.mcBase import RNG


torch.set_printoptions(12)

# Specify simulation dimensions
N = 1024
T = 1.0
delta = 0.25
use_av = True

t = torch.linspace(start=delta, end=T, steps=int(T/delta))
M = len(t) - 1

# Create model
a = torch.tensor(0.86)
b = torch.tensor(0.08)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)
model = Vasicek(a, b, sigma)
swap_rate = model.calc_swap_rate(r0, t, delta)


# RNG
seed = None
rng = RNG(N, seed, use_av)

# Initialize object for storing variables
r = torch.full(size=(M + 1, N), fill_value=torch.nan)
payoff = torch.full((M + 1, N), torch.nan)
df = model.calc_zcb(r0, t)

r[0, :] = r0
payoff[0, :] = 0.0

# First step, cap pays nothing


# Simulate
for k in range(0, M):
    Z = rng.next_G()
    r[k + 1, :] = model.simulate(r0=r[k, :], Z=Z, dt=delta)
    F = model.calc_fwd(r0=r[k+1, :], t=0.0, delta=delta)
    payoff[k+1, :] = delta * torch.maximum(F - swap_rate, torch.tensor(0.0))

cap_mc = torch.mean(torch.sum(df[:, None] * payoff, dim=0))
cap_model = model.calc_cap(r0=r0, t=t, delta=delta, K=swap_rate)

print(cap_mc, cap_model)
