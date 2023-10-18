import torch
from application.engine.vasicek import Vasicek
from application.engine.black import black_cpl, black_cpl_iv, black_cpl_delta
from application.engine.mcBase import mcSimPaths, RNG
from application.engine.products import Caplet
from application.utils.torch_utils import max0
import matplotlib.pyplot as plt

"""
    NOTE: This is script is very slow as no optimization has been made.
    The to estimate `delta` the script solves for the Black76 implied volatility
    using the analytical pricing formula in the Vasicek model as the `market price`
    of the caplet.
"""


def calc_black_delta_bump_and_reval(cpl, zcb, fwd, strike, tau, delta, bump=torch.tensor(0.0001)):
    sigma_black = black_cpl_iv(market_price=cpl, zcb=zcb, fwd=fwd, K=strike, t=tau, delta=delta)

    # Estimate delta using central finite difference
    cpl_black = black_cpl(sigma_black, zcb=zcb, fwd=fwd - 0.5 * bump, K=strike, t=tau, delta=delta)
    cpl_black_bump = black_cpl(sigma_black, zcb=zcb, fwd=fwd + 0.5 * bump, K=strike, t=tau, delta=delta)
    delta = (cpl_black_bump - cpl_black) / bump
    return delta


def calc_black_delta_analytical(cpl, zcb, fwd, strike, tau, delta):
    sigma_black = black_cpl_iv(market_price=cpl, zcb=zcb, fwd=fwd, K=strike, t=tau, delta=delta)
    delta = black_cpl_delta(sigma_black=sigma_black, zcb=zcb, fwd=fwd, K=strike, t=tau, delta=delta)
    return delta


torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = None

    N = 1024

    measure = 'risk_neutral'

    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)

    start = torch.tensor(0.25)
    delta = torch.tensor(0.25)

    model = Vasicek(a, b, sigma, r0, False, False, measure)

    strike = model.calc_swap_rate(r0, start.view(1), delta)

    rng = RNG(seed=seed, use_av=True)

    prd = Caplet(
        strike=strike,
        start=start,
        delta=delta
    )

    """
        HEDGE EXPERIMENT
    """

    M = 25

    hedgeTL = torch.linspace(0.0, float(start), M+1)
    mcSimPaths(prd, model, rng, N, hedgeTL)
    r = model.x

    cpl0 = model.calc_cpl(r0, start, delta, strike)[0]
    zcb = model.calc_zcb(r0, start + delta)[0]
    fwd = model.calc_fwd(r0, start, delta)[0]

    # Initialize experiments
    V = cpl0 * torch.ones(size=(N, ))
    #a = calc_black_delta_bump_and_reval(cpl0, zcb, fwd, strike, start, delta) * torch.ones(size=(N, ))
    a = calc_black_delta_analytical(cpl0, zcb, fwd, strike, start, delta) * torch.ones(size=(N, ))
    b = V - a * fwd * torch.ones(size=(N, ))
    payoff = torch.full_like(V, torch.nan)
    fwdT = torch.full_like(V, torch.nan)

    for k in range(1, len(hedgeTL)):            # Loop over time
        dt = hedgeTL[k] - hedgeTL[k - 1]
        t = hedgeTL[k]
        tau = start - t
        for p in range(N):                      # Loop over paths
            # Simulation step
            rs = r[k, p]

            # Update market variables
            zcb = model.calc_zcb(rs, tau)[0]
            fwd = model.calc_fwd(rs, tau, delta)[0]

            # Update portfolio
            V[p] = a[p] * fwd + b[p] * torch.exp(0.5 * (r[k, p] + r[k-1, p]) * dt)

            if k < len(hedgeTL) - 1:
                cpl = model.calc_cpl(r[k, p], tau, delta,
                                     strike)  # This might be cheating, but necessary to calculate a

                #a[p] = calc_black_delta_bump_and_reval(cpl, zcb, fwd, strike, tau, delta)
                a[p] = calc_black_delta_analytical(cpl, zcb, fwd, strike, tau, delta)
                b[p] = V[p] - a[p] * fwd
            else:
                fwdT[p] = fwd

    payoff = delta * max0(fwdT - strike)

    plt.figure()
    plt.plot(fwdT,  payoff, 'o', color='red', label='Payoff', alpha=0.2)
    plt.plot(fwdT, V, 'o', color='blue', label='Hedge', alpha=0.2)
    plt.title('Replication of payoff function')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(fwdT, a, 'o', alpha=0.2, color='gray', label='Delta(T-1) vs Fwd(T)')
    plt.title('Black Delta(T-1) as a function of Fwd(T)')
    plt.legend()
    plt.xticks((min(fwdT), strike, max(fwdT)))
    plt.show()
