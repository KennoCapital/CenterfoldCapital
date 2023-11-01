from application.engine.vasicek import Vasicek
from application.engine.products import EuropeanPayerSwaption, EuropeanReceiverSwaption
from application.engine.mcBase import mcSim, RNG
import torch

torch.set_default_dtype(torch.float64)

"""
Testing put-call parity for swaptions.
"""
if __name__ == '__main__':

    seed = None

    measure = 'terminal'
    N = 50000

    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)

    strike = torch.tensor(0.092)

    exerciseDate = torch.tensor(5.0)
    delta = torch.tensor(0.5)
    swapFirstFixingDate = torch.tensor(6.0)
    swapLastFixingDate = torch.tensor(30.0)
    notional = torch.tensor(1E6)

    if measure == 'risk_neutral':
        dTL = torch.linspace(0.0, float(exerciseDate), int(50 * exerciseDate) + 1)
    else:
        dTL = torch.tensor([])

    rng = RNG(seed=seed, use_av=True)

    mdl = Vasicek(a=a, b=b, sigma=sigma, r0=r0, use_ATS=False, use_euler=False, measure=measure)

    t = torch.linspace(float(swapFirstFixingDate),
                       float(swapLastFixingDate),
                       int((swapLastFixingDate - swapFirstFixingDate) / delta + 1))

    swpt_payer = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    swpt_receiver = EuropeanReceiverSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    price_swpt_payer = torch.mean(mcSim(swpt_payer, mdl, rng, N, dTL))
    price_swpt_receiver = torch.mean(mcSim(swpt_receiver, mdl, rng, N, dTL))
    price_swap = mdl.calc_swap(r0, t, delta, strike, notional)

    print(f'SwptPayer - SwptReceiver\n'
          f' = {price_swpt_payer} - {price_swpt_receiver}\n'
          f' = {price_swpt_payer-price_swpt_receiver}')

    print(f'SwapPayer = {price_swap}')

    diff_nom = price_swpt_payer - price_swpt_receiver - price_swap
    diff_rel = diff_nom / notional

    print(f'Difference = {diff_nom}')
    print(f'Difference relative to notional = {diff_rel}')
