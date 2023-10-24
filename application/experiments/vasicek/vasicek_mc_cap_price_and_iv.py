from application.engine.vasicek import *
from application.engine.black import black_cpl_iv
from application.engine.products import Cap
from application.engine.mcBase import mcSim, RNG

"""
    This script replicates the results in Filipovic's table 7.1
"""

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)

if __name__ == '__main__':
    delta = torch.tensor(0.25)
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'terminal'
    seed = 1234
    use_av = True

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True)
    rng = RNG(seed=seed, use_av=use_av)

    N = 50000

    for T in torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]):
        t = torch.linspace(start=float(delta), end=float(T), steps=int(T / delta))

        # Calculate ATM cap price
        # Note the slice on `t`. The last time is not included to match the experiment in Filipovic Table 7.1
        swap_rate = mdl.calc_swap_rate(r0, t[:-1], delta)

        prd = Cap(
            strike=swap_rate,
            firstFixingDate=torch.tensor(0.25),
            lastFixingDate=torch.tensor(T) - delta,
            delta=delta
        )

        cap = torch.mean(torch.sum(mcSim(prd, mdl, rng, N), dim=0))

        # Calculate ATM Implied Volatility
        zcb = mdl.calc_zcb(r0, t).flatten()
        fwd = (zcb[:-1] / zcb[1:] - 1) / delta
        sigma_iv = black_cpl_iv(cap, zcb=zcb[1:], fwd=fwd, K=swap_rate, t=t[:-1], delta=delta)

        print(f'T = {T}\t|\tCapATM(T) = {cap}\t|\t BlackATMvol = {sigma_iv}')
