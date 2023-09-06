import torch
import matplotlib.pyplot as plt
from application.engine.vasicek import Vasicek
from application.engine.mcBase import RNG


# Specify simulation dimensions
N = 4
t = torch.linspace(start=0.25, end=30.0, steps=120)
dt = t[1:] - t[:-1]
M = len(t) - 1

# Create model
a = torch.tensor(0.86)
b = torch.tensor(0.08)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)
model = Vasicek(a, b, sigma)

# RNG
seed = 1234
rng = RNG(N, seed)

# Initialize object for storing variables
r = torch.full(size=(M + 1, N), fill_value=torch.nan)
r[0, :] = r0

# Simulate
for k in range(M):
    Z = rng.next_G()
    r[k+1, :] = model.simulate(r0=r[k, :], dt=dt[k], Z=Z)

plt.plot(t, r, alpha=0.5)
plt.show()

