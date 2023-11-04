from application.engine.mcBase import lsmcDefaultSim, LSMC, RNG, mcSim
from application.engine.products import BermudanPayerSwaption, EuropeanPayerSwaption
from application.engine.vasicek import Vasicek
from application.engine.regressor import PolynomialRegressor
import torch

torch.set_printoptions(2)
torch.set_default_dtype(torch.float64)

"""
Testing implementation of Bermudan swaption and computes naive upper and lower bound.
"""

if __name__ == '__main__':
    seed = 1234

    use_SVD = True
    bias = True
    include_interactions = True
    deg = 5

    n = 5000
    N = 50000

    measure = 'terminal'

    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)

    exerciseDates = torch.tensor([5.0, 10.0, 15.0])
    delta = torch.tensor(0.25)
    swapLastFixingDate = torch.tensor(30.0)
    notional = torch.tensor(1e6)

    if measure == 'risk_neutral':
        dTL = torch.linspace(0.0, float(exerciseDates[-1]), 50 * int(exerciseDates[-1]) + 1)
    else:
        dTL = torch.tensor([])

    model = Vasicek(a, b, sigma, r0, False, False, measure)

    t = torch.linspace(float(exerciseDates[-1]),
                       float(swapLastFixingDate),
                       int((swapLastFixingDate - exerciseDates[-1]) / delta) + 1)
    strike = model.calc_swap_rate(r0, t, delta)

    rng = RNG(seed=seed, use_av=True)

    # Bermudan Payer Swaption
    bermudan_payer_swpt = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    poly_reg = PolynomialRegressor(deg=deg, use_SVD=use_SVD, bias=bias, include_interactions=include_interactions)
    lsmc = LSMC(reg=poly_reg)

    payoff = lsmcDefaultSim(
        prd=bermudan_payer_swpt, mdl=model, rng=rng, N=N, n=n, lsmc=lsmc, reg=poly_reg, dTL=dTL
    )

    price_bermudan_payer_swpt = torch.mean(torch.sum(payoff, dim=0))

    # Determine upper and lower bound
    lower_bound = torch.tensor(0.0)
    upper_bound = torch.tensor(0.0)
    eu_prices = []

    for T in exerciseDates:
        european_payer_swpt = EuropeanPayerSwaption(
            strike=strike,
            exerciseDate=T,
            delta=delta,
            swapFirstFixingDate=T,
            swapLastFixingDate=swapLastFixingDate,
            notional=notional
        )

        payoff = mcSim(prd=european_payer_swpt, mdl=model, rng=rng, N=N, dTL=dTL)
        eu_prices.append(torch.mean(payoff).view(1))

    eu_prices = torch.concat(eu_prices)
    lower_bound = torch.max(eu_prices)
    upper_bound = torch.sum(eu_prices)

    print(f'EuropeanPayerSwpt (lower bound) = {lower_bound}')
    print(f'BermudanPayerSwpt = {price_bermudan_payer_swpt}')
    print(f'EuropeanPayerSwpt (upper bound) = {upper_bound}')
