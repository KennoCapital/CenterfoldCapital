from application.engine.vasicek import *
from application.engine.black import black_cpl_iv


if __name__ == '__main__':
    torch.set_printoptions(precision=12)
    T = 1.0
    delta = 0.25
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    t = torch.linspace(start=delta, end=T, steps=int(T / delta))

    # Calculate ATM cap price
    # Note the slice on `t`. The last time is not included to match the experiment in Filipovic Table 7.1
    mld = Vasicek(a, b, sigma, use_ATS=False)
    swap_rate = mld.calc_swap_rate(r0, t, delta)
    cap = mld.calc_cap(r0, t[:-1], delta, swap_rate)
    print(cap)

    # Calculate ATM Implied Volatility
    zcb = mld.calc_zcb(r0, t)
    fwd = (zcb[:-1] / zcb[1:] - 1) / delta

    sigma_iv = black_cpl_iv(cap, zcb=zcb[1:], fwd=fwd, K=swap_rate, t=t[:-1], delta=delta)
    print(sigma_iv)
