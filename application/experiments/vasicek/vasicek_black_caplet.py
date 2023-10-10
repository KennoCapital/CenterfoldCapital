from application.engine.vasicek import *
from application.engine.black import black_cpl_iv, black_cpl


if __name__ == '__main__':
    torch.set_printoptions(precision=12)
    T = 1.0
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
    print(f'Vasicek cap price: {cap}')

    cpl = mld.calc_cpl(r0=r0, t=t, delta=delta, K=swap_rate)

    # Calculate ATM Implied Volatility
    zcb = mld.calc_zcb(r0, t)
    fwd = (zcb[:-1] / zcb[1:] - 1) / delta

    sigma_iv = black_cpl_iv(market_price=cap, zcb=zcb[:-1], fwd=fwd, K=swap_rate, t=t[:-1], delta=delta)
    print(f'Black sigma IV: {sigma_iv}')

    # Calculate cap price using Black's formula
    clps_black = torch.tensor(
        [black_cpl(sigma=sigma_iv, zcb=zcb[i], fwd=fwd[i], K=swap_rate, t=s, delta=delta)
         for i, s in enumerate(t[:-1])]
    )

    cap_black = torch.sum(clps_black)
    print(f'Black cap price: {cap_black}')

