from application.engine.mcBase import lsmcDefaultSim, LSMC, RNG, mcSim
from application.engine.products import BermudanPayerSwaption, EuropeanPayerSwaption
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.regressor import PolynomialRegressor
from application.engine.products import Caplet
import torch
import matplotlib.pyplot as plt

torch.set_printoptions(2)
torch.set_default_dtype(torch.float64)

"""
Testing implementation of MC simulation and engine.
"""

if __name__ == '__main__':
    seed = 1234

    measure = 'risk_neutral'

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.5509) #0553
    sigma = torch.tensor(1.0497) #3325
    alpha0 = torch.tensor(0.0001) #45
    alpha1 = torch.tensor(0.0046) #0131
    gamma = torch.tensor(0.1777)
    rho = torch.tensor(0.327)
    theta = torch.tensor(2.1070) #* torch.tensor(.1476)/ kappa
    varphi = torch.tensor(0.068)

    # Product specification
    start = torch.tensor(15.0)
    delta = torch.tensor(.25)
    strike = torch.tensor(0.084)
    notional = torch.tensor(1e6)

    dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

    # instantiate model
    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi)

    rng = RNG(seed=seed, use_av=True)

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

    # plotting state vars
    x,v,phi1,phi2,phi3,phi4,phi5,phi6 = [i.squeeze()[:,0] for i in model.x]

    fig1, axs1 = plt.subplots(1, 2, figsize=(8, 4))
    axs1[0].plot(dTL, x[:-1], label = r'$x$')
    axs1[0].legend(loc='upper left')
    axs1[1].plot(dTL, v[:-1], label=r'$\nu$')
    axs1[1].legend(loc='upper left')

    fig2, axs2 = plt.subplots(3, 2, figsize=(7, 8))
    axs2[0, 0].plot(dTL, phi1[:-1], label=r'$\phi_1$')
    axs2[0, 0].legend(loc='upper left')
    axs2[0, 1].plot(dTL, phi2[:-1], label=r'$\phi_2$')
    axs2[0, 1].legend(loc='upper left')
    axs2[1, 0].plot(dTL, phi3[:-1], label=r'$\phi_3$')
    axs2[1, 0].legend(loc='upper left')
    axs2[1, 1].plot(dTL, phi4[:-1], label=r'$\phi_4$')
    axs2[1, 1].legend(loc='upper left')
    axs2[2, 0].plot(dTL, phi5[:-1], label=r'$\phi_5$')
    axs2[2, 0].legend(loc='upper left')
    axs2[2, 1].plot(dTL, phi6[:-1], label=r'$\phi_6$')
    axs2[2, 1].legend(loc='upper left')

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(dTL, model._fwd_rate_vol(0,dTL) )
    plt.show()