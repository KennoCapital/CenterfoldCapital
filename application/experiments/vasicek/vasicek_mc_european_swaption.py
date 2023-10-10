from application.engine.mcBase import mcSim, RNG
from application.engine.products import EuropeanPayerSwaption
from application.engine.vasicek import Vasicek
import torch
import matplotlib.pyplot as plt


torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    seed = 1234

    rep = 100
    N_list = [2 ** i for i in range(10, 21)]  # ~1k to 1m

    measure = 'terminal'

    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)

    exerciseDate = torch.tensor(5.0)
    delta = torch.tensor(0.5)
    swapFirstFixingDate = torch.tensor(6.0)
    swapLastFixingDate = torch.tensor(30.0)
    notional = torch.tensor(1e6)

    t = torch.linspace(float(swapFirstFixingDate),
                       float(swapLastFixingDate),
                       int((swapLastFixingDate - swapFirstFixingDate) / delta + 1))

    if measure == 'risk_neutral':
        dTL = torch.linspace(0.0, float(exerciseDate), int(50 * exerciseDate) + 1)
    else:
        dTL = torch.tensor([])

    model = Vasicek(a, b, sigma, r0, False, use_euler=False, measure=measure)
    swap_rate = model.calc_swap_rate(r0, t, delta)
    rng = RNG(seed=seed, use_av=True)

    prd = EuropeanPayerSwaption(
        strike=swap_rate,
        exerciseDate=exerciseDate,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        delta=delta,
        notional=notional
    )


    def priceMC(N):
        cashflows = mcSim(prd, model, rng, N, dTL)
        price = torch.mean(cashflows)
        return price


    res = torch.tensor([[priceMC(N) for N in N_list] for _ in range(rep)])

    # Plotting
    plt.plot(N_list, torch.std(res / notional, dim=0))
    plt.ylabel('std(Price / Notional)')
    plt.xlabel('N')
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.xticks(N_list, labels=N_list, rotation=45)
    plt.show()
