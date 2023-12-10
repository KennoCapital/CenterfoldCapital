import torch
from torch.func import jacfwd
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.mcBase import mcSim, RNG
from application.engine.products import EuropeanPayerSwaption
from functools import partial

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


def calc_swpt(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi,
              prd, rng, N, dTL, simDim):

    cPrd = EuropeanPayerSwaption(
        strike=prd.strike,
        exerciseDate=prd.exerciseDate,
        delta=prd.delta,
        swapFirstFixingDate=prd.swapFirstFixingDate,
        swapLastFixingDate=prd.swapLastFixingDate,
        notional=prd.notional
    )
    cRng = RNG(seed=rng.seed, use_av=rng.use_av)
    cMdl = trolleSchwartz(
        v0=v0,
        gamma=gamma,
        kappa=kappa,
        theta=theta,
        rho=rho,
        sigma=sigma,
        alpha0=alpha0,
        alpha1=alpha1,
        varphi=varphi,
        simDim=simDim
    )
    cTL = dTL
    payoff = mcSim(cPrd, cMdl, cRng, N, cTL)
    price = torch.mean(payoff)
    return price.view(1), payoff


if __name__ == '__main__':
    seed = 1234
    N = 50000
    steps_per_year = 100
    use_av = True

    # Model specification
    kappa = torch.tensor(0.0553) 
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045) 
    alpha1 = torch.tensor(0.131) 
    gamma = torch.tensor(0.3341) 
    rho = torch.tensor(0.4615) 
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476) 
    v0 = theta
    varphi = torch.tensor(0.0832)

    param = v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi

    mdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    strike = torch.tensor(0.08)
    delta = torch.tensor(0.25)
    swapFirstFixingDate = torch.tensor(1.0)
    swapLastFixingDate = torch.tensor(5.0) + exerciseDate
    notional = torch.tensor(1e6)

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    # Monte Carlo specification
    rng = RNG(seed=seed, use_av=use_av)
    dTL = torch.linspace(0.0, float(exerciseDate), int(exerciseDate * steps_per_year) + 1)

    # Auxiliary function
    V = partial(calc_swpt, prd=prd, rng=rng, N=N, dTL=dTL, simDim=1)

    # Evaluate function and its Jacobian
    price, payoff = V(*param)
    J = jacfwd(V, argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8), randomness='same')(*param)

    dPrice_dParam = torch.vstack(J[0])
    dPayoff_dParam = torch.vstack(J[1])
