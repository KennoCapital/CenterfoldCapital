from application.engine.mcBase import mcSim, RNG
from application.engine.products import Caplet
from application.engine.vasicek import Vasicek
import torch

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = None

    N = 50000

    measure = 'terminal'

    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    notional = torch.tensor(1e6)

    start = torch.tensor(5.0)
    delta = torch.tensor(1.0)

    dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

    model = Vasicek(a, b, sigma, r0, True, False, measure)
    swap_rate = torch.tensor(0.084)

    rng = RNG(seed=seed, use_av=True)

    prd = Caplet(
        strike=swap_rate,
        start=start,
        delta=delta,
        notional=notional
    )

    payoff = mcSim(prd, model, rng, N, dTL)

    print(
        'MC price =', torch.mean(payoff)
    )

    print(
        'Analytical price =', model.calc_cpl(r0, start, delta, swap_rate, notional)[0][0]
    )
