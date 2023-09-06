from application.engine.vasicek import *
from application.engine.black import black_iv


if __name__ == '__main__':
    torch.set_printoptions(precision=12)
    T = 30.0
    delta = 0.25
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    n = int(T / delta - 1)
    t = torch.linspace(start=delta, end=T, steps=n + 1)

    # Calculate ATM cap price
    mld = Vasicek(a, b, sigma, use_ATS=True)
    swap_rate = mld.calc_swap_rate(r0, t, delta)
    cap = mld.calc_cap(r0, t, delta, swap_rate)
    print(cap)

    # Calculate ATM Implied Volatility
    zcb = mld.calc_zcb(r0, t)
    fwd = (zcb[:-1] / zcb[1:] - 1) / delta

    sigma_iv = black_iv(cap, zcb=zcb[:-1], fwd=fwd, K=swap_rate, t=t[:-1], delta=delta)
    print(sigma_iv)
