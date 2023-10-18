import torch
import matplotlib.pyplot as plt
from application.engine.vasicek import Vasicek
from application.engine.mcBase import RNG

# structure:
# 1) simulate state variable(s)
# 2) compute ZCB structure
# 3) calculate underlying libor rate/underlying to price derivatives


if __name__ == '__main__':
    # Specify simulation dimensions
    N = 10
    T = 1
    delta = 0.25
    use_av = True

    t = torch.linspace(start=delta, end=T, steps=int(T / delta))
    M = len(t) - 1

    # Create model
    a = torch.tensor(0.86)
    b = torch.tensor(0.08)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    model = Vasicek(a, b, sigma)


    # RNG
    seed = 1234
    rng = RNG(N, seed, use_av)

    # Initialize object for storing variables
    r = torch.full(size=(M + 1, N), fill_value=torch.nan)
    ZCB = torch.full(size=(M, N), fill_value=torch.nan)

    r[0, :] = r0

    # Simulate
    for k in range(0, M):
        Z = torch.normal(0.0, 1.0, size=(1, N))
        r[k + 1, :] = model.simulate(r0=r[k, :], Z=Z, dt=delta)
        ZCB[k, :] = model.calc_zcb(r[k + 1], t[k + 1])

    swaprate = model.calc_swap_rate(r0, t, delta)
    FWD = (1 / ZCB - 1) / delta
    discount = model.calc_zcb(r0, t[1:])
    payoff = torch.sum(torch.t(torch.maximum(FWD - swaprate, torch.tensor(0.0))
                               )
                    * discount, axis = 0)
    print(f"pathwise payoffs for caps are {payoff}")




