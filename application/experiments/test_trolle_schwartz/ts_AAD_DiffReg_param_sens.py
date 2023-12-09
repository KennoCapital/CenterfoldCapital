import torch
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.mcBase import mcSim, RNG
from application.engine.products import EuropeanPayerSwaption

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

# TODO find a fast way to calculate the Jacobian for all the parameters at the same time

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

    torch.autograd.set_detect_anomaly(True)
    v0.requires_grad = True

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

    payoff = mcSim(prd, mdl, rng, N, dTL)

    swpt = torch.mean(payoff)
    swpt.backward(retain_graph=True)
    print(f'Swpt={swpt}, v0_grad={v0.grad}')
