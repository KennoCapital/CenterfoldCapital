from application.engine.products import CapletAsPutOnZCB
from application.engine.trolleSchwartz import trolleSchwartz
import torch
from application.engine.mcBase import mcSim, RNG


torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    euler_step_size = 100

    # Setup
    seed = 1234
    N = 50000
    measure = 'risk_neutral'

    # Trolle-Schwartz model specification
    # ref to paper for values
    v0 = torch.tensor([9.34233402, 1.19091676, 1.90420149])
    kappa = torch.tensor([0.2169, 0.5214, 0.8340])
    sigma = torch.tensor([0.6586, 1.0212, 1.2915])
    alpha0 = torch.tensor([0.0000, 0.0014, -0.0085])
    alpha1 = torch.tensor([0.0037, 0.0320, 0.0272])
    gamma = torch.tensor([0.1605, 1.4515, 0.6568])
    rho = torch.tensor([0.0035, 0.0011, 0.6951])
    thetaP = torch.tensor([1.4235, 0.7880, 1.2602])

    kappaP = torch.tensor([1.4235, 0.7880, 1.2602])

    theta = thetaP * kappaP / kappa

    # initialize IFR
    varphi = torch.tensor(0.0668)

    # Product specification
    exerciseDate = torch.tensor(2.0)
    delta = torch.tensor(.25)
    strike = torch.tensor(0.07)
    notional = torch.tensor(1e6)
    model = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=3)

    # Analytical solution
    cpl = model.calc_cpl(torch.tensor(0.), exerciseDate, delta, strike, notional)
    print('Semi-analytic Price =', cpl)

    # Monte Carlo simulation
    dTL = torch.linspace(0.0, exerciseDate + delta, int(euler_step_size * (exerciseDate + delta) + 1))

    rng = RNG(seed=seed, use_av=True)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    payoff = mcSim(prd, model, rng, N, dTL)

    mc_price = torch.mean(payoff)
    print('MC Price =', mc_price)



