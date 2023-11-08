from application.engine.products import EuropeanPayerSwaption
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.mcBase import mcSim, RNG
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    # Setup
    seed = 1234
    rep = 50
    N_list = [2 ** i for i in range(10, 16)]  # ~1k to 65k
    measure = 'risk_neutral'

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553)
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542)
    varphi = torch.tensor(0.0832)

    # Product specification
    exerciseDate = torch.tensor(5.0)
    delta = torch.tensor(0.5)
    swapFirstFixingDate = torch.tensor(6.0)
    swapLastFixingDate = torch.tensor(30.0)
    notional = torch.tensor(1e6)

    t = torch.linspace(float(swapFirstFixingDate),
                       float(swapLastFixingDate),
                       int((swapLastFixingDate - swapFirstFixingDate) / delta + 1))


    dTL = torch.linspace(0.0, float(exerciseDate), int(50 * exerciseDate) + 1)

    # instantiate model
    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi)
    swap_rate = torch.tensor(0.084) #model.calc_swap_rate([x[:,0,:].mean(1) for x in model.x], t, delta)
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


    res = torch.tensor([[priceMC(N) for N in N_list] for _ in tqdm(range(rep))])

    # Plotting
    plt.plot(N_list, torch.std(res / notional, dim=0))
    plt.ylabel('std(Price / Notional)')
    plt.xlabel('N')
    plt.yscale('log', base=2)
    plt.xscale('log', base=2)
    plt.xticks(N_list, labels=N_list, rotation=45)
    plt.show()

