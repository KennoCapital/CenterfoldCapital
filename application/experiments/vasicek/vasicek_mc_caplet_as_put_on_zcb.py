import torch
from application.engine.products import CapletAsPutOnZCB, Caplet
from application.engine.mcBase import mcSim, RNG
from application.engine.vasicek import Vasicek

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'
    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    # RNG specification
    seed = 1234
    use_av = True
    N = 500000
    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(0.25)
    delta = torch.tensor(0.25)
    strike = mdl.calc_swap_rate(r0, exerciseDate, delta)

    prd1 = CapletAsPutOnZCB(
        exerciseDate=exerciseDate,
        strike=strike,
        delta=delta
    )

    prd2 = Caplet(strike=strike, start=exerciseDate, delta=delta)

    # Timeline
    dTL = torch.linspace(0.0, float(exerciseDate), 50 * int(exerciseDate) + 1)

    # Simulation
    cpl1 = torch.mean(mcSim(prd1, mdl, rng, N, dTL))
    cpl2 = torch.mean(mcSim(prd2, mdl, rng, N, dTL))

    # Analytical
    cpl3 = mdl.calc_cpl(r0, exerciseDate, delta, strike)[0][0]

    print(f'Caplet price:\n'
          f'as Put Option on ZCB \t: {cpl1}\n'
          f'as Call Option on Fwd\t: {cpl2}\n'
          f'as analytical:       \t: {cpl3}')
