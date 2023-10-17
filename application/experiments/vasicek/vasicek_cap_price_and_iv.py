from application.engine.vasicek import *
from application.engine.black import black_cpl_iv

"""
    This script replicates the results in Filipovic's table 7.1
"""

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)

if __name__ == '__main__':

    for T in torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]):
        delta = torch.tensor(0.25)
        a = torch.tensor(0.86)
        b = torch.tensor(0.09)
        sigma = torch.tensor(0.0148)
        r0 = torch.tensor(0.08)
        t = torch.linspace(start=float(delta), end=float(T), steps=int(T / delta))

        # Calculate ATM cap price
        # Note the slice on `t`. The last time is not included to match the experiment in Filipovic Table 7.1
        mld = Vasicek(a, b, sigma, use_ATS=True)
        swap_rate = mld.calc_swap_rate(r0, t[:-1], delta)
        cap = mld.calc_cap(r0, t[:-1], delta, swap_rate)

        # Calculate ATM Implied Volatility
        zcb = mld.calc_zcb(r0, t).flatten()
        fwd = (zcb[:-1] / zcb[1:] - 1) / delta
        sigma_iv = black_cpl_iv(cap, zcb=zcb[1:], fwd=fwd, K=swap_rate, t=t[:-1], delta=delta)

        print(f'T = {T}\t|\tCapATM(T) = {cap}\t|\t BlackATMvol = {sigma_iv}')
