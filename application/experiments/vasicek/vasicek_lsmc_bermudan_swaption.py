from application.engine.mcBase import lsmcDefaultSim, LSMC, RNG, mcSim
from application.engine.products import BermudanPayerSwaption, EuropeanPayerSwaption
from application.engine.vasicek import Vasicek
from application.engine.regressor import PolynomialRegressor
import torch

torch.set_printoptions(2)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    seed = 1234

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
    swapFirstFixingDate = torch.tensor(15.0)
    swapLastFixingDate = torch.tensor(30.0)
    # strike = torch.tensor(0.09102013)
    notional = torch.tensor(1e6)

    if measure == 'risk_neutral':
        dTL = torch.linspace(0.0, float(exerciseDates[-1]), 50 * int(exerciseDates[-1]) + 1)
    else:
        dTL = torch.tensor([])

    model = Vasicek(a, b, sigma, r0, False, False, measure)

    t = torch.linspace(float(swapFirstFixingDate),
                       float(swapLastFixingDate),
                       int((swapLastFixingDate - swapFirstFixingDate) / delta) + 1)
    strike = model.calc_swap_rate(r0, t, delta)

    rng = RNG(seed=seed, use_av=True)

    # Bermudan Payer Swaption
    bermudan_payer_swpt = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    poly_reg = PolynomialRegressor(deg=deg, standardize=True)
    lsmc = LSMC(reg=poly_reg)

    payoff = lsmcDefaultSim(
        prd=bermudan_payer_swpt, mdl=model, rng=rng, N=N, n=n, lsmc=lsmc, reg=poly_reg, dTL=dTL
    )

    price_bermudan_payer_swpt = torch.mean(torch.sum(payoff, dim=0))

    print(f'BermudanPayerSwpt = {price_bermudan_payer_swpt}')

    # European Payer Swaption (lower bound)
    european_payer_swpt = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDates[-1],
        delta=delta,
        swapFirstFixingDate=exerciseDates[-1],
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    payoff = mcSim(prd=european_payer_swpt, mdl=model, rng=rng, N=N, dTL=dTL)
    price_european_payer_swpt = torch.mean(payoff)

    print(f'EuropeanPayerSwpt (lower bound) = {price_european_payer_swpt}')

    # Sum of European Payer Swaption (upper bound)
    upper_bound = 0.0
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
        upper_bound += torch.mean(payoff)

    print(f'EuropeanPayerSwpt (upper bound) = {upper_bound}')
