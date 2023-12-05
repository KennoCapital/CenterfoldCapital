from application.engine.products import Caplet
from application.engine.trolleSchwartz import trolleSchwartz
import torch
from application.engine.mcBase import mcSim, RNG
from tqdm.contrib import itertools
import matplotlib.pyplot as plt

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    strike_plot = False
    # Setup
    seed = 1234
    N = 1024*10
    measure = 'risk_neutral'

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553)
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476)
    #
    varphi = torch.tensor(0.0832)

    # Product specification
    start = torch.tensor(1.0)
    delta = torch.tensor(.25)
    strike = torch.tensor(.09)
    notional = torch.tensor(1e6)

    dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

    # instantiate model
    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi)

    rng = RNG(seed=seed, use_av=True)

    prd = Caplet(
        strike=strike,
        start=start,
        delta=delta,
        notional=notional
    )

    cashflows = mcSim(prd, model, rng, N, dTL)
    payoff = torch.sum(cashflows, dim=0)

    # mc
    mc_price = torch.nanmean(payoff)
    print('MC Price =', mc_price)

    # analytic
    #cpl = model.calc_cpl(0, prd.start, prd.delta, prd.strike, notional)
    #print('Semi-analytic Price =', cpl)


    if strike_plot:
        strikes = torch.linspace(0.025, 0.14, 10)
        times = torch.tensor([1., 2.5, 5.])

        prices = torch.empty(len(strikes) * len(times))
        mc_prices = torch.empty(len(strikes) * len(times))
        i = 0

        for t, s in itertools.product(times, strikes):
            prd = Caplet(
                strike=s,
                start=t,
                delta=delta,
                notional=notional
            )
            cashflows = mcSim(prd, model, rng, N, dTL)
            payoff = torch.sum(cashflows, dim=0)

            # mc
            mc_price = torch.nanmean(payoff)

            # analytical
            cpl = model.calc_cpl(0, prd.start, prd.delta, prd.strike, notional)

            prices[i] = cpl
            mc_prices[i] = mc_price
            i += 1

        prices1 = prices[:10]
        prices2 = prices[10:20]
        prices3 = prices[20:31]

        mcprices1 = mc_prices[:10]
        mcprices2 = mc_prices[10:20]
        mcprices3 = mc_prices[20:31]

        plt.figure()
        plt.plot(strikes, prices1, label='1')
        plt.plot(strikes, prices2, label='2.5')
        plt.plot(strikes, prices3, label='5.')
        plt.xlabel('strike')
        plt.ylabel('price')
        plt.legend()
        plt.title('TS cpl price' + f' N = {notional} with delta = {delta}')
        plt.show()

        plt.figure()
        plt.plot(strikes, mcprices1, label='1')
        plt.plot(strikes, mcprices2, label='2.5')
        plt.plot(strikes, mcprices3, label='5.')
        plt.xlabel('strike')
        plt.ylabel('price')
        plt.legend()
        plt.title('TS cpl MC price' + f' N = {notional} with delta = {delta}')
        plt.show()


