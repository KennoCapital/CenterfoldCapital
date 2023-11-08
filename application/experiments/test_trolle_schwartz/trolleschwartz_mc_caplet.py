from application.engine.products import Caplet
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.vasicek import Vasicek
import torch
from application.engine.mcBase import mcSim, RNG
import matplotlib.pyplot as plt

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    # Setup
    seed = 1234
    N = 1024*10
    measure = 'risk_neutral'
    produce_plots = False
    perform_calibration = False
    gradient_plot = True

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553) #0553
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045) #045
    alpha1 = torch.tensor(0.131) #131
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542) #7542
    #
    varphi = torch.tensor(0.0832)

    # Product specification
    start = torch.tensor(1.0)
    delta = torch.tensor(.25)
    strike = torch.tensor(0.084)
    notional = torch.tensor(1.0)

    dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

    # instantiate model
    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi)

    rng = RNG(seed=seed, use_av=False)

    prd = Caplet(
        strike=strike,
        start=start,
        delta=delta,
        notional=notional
    )

    cashflows = mcSim(prd, model, rng, N, dTL)
    payoff = torch.sum(cashflows, dim=0)

    # mc
    mc_price = torch.nanmean(payoff)
    print('MC Price =', mc_price)

    # analytic
    cpl = model.calc_cpl(0, prd.start, prd.delta, prd.strike, notional)
    print('Semi-analytic Price =', cpl)
