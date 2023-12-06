from application.engine.products import Caplet
from application.engine.trolleSchwartz import trolleSchwartz
import torch
from application.engine.mcBase import mcSim, RNG
import matplotlib.pyplot as plt
from tqdm.contrib import itertools
from application.utils.path_config import get_plot_path


torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    strike_plot = True
    save_fig = True
    euler_step_size = 100

    # Setup
    seed = 1234
    N = 1024*10
    measure = 'risk_neutral'

    # Trolle-Schwartz model specification
    # ref to paper for values
    kappa = torch.tensor([0.2169, 0.5214, 0.8340])
    sigma = torch.tensor([0.6586, 1.0212, 1.2915])
    alpha0 = torch.tensor([0.0000, 0.0014, -0.0085])
    alpha1 = torch.tensor([0.0037, 0.0320, 0.0272])
    gamma = torch.tensor([0.1605, 1.4515, 0.6568])
    rho = torch.tensor([0.0035, 0.0011, 0.6951])
    thetaP = torch.tensor([1.4235, 0.7880, 1.2602])

    kappaP = torch.tensor([1.4235, 0.7880, 1.2602])

    theta = thetaP * kappaP / kappa

    # initialize IFR
    varphi = torch.tensor(0.0668)

    # Product specification
    start = torch.tensor(5.)
    delta = torch.tensor(.25)
    strike = torch.tensor(0.07)
    notional = torch.tensor(1e6)

    dTL = torch.linspace(0.0, start + delta, int(euler_step_size * (start + delta) + 1))

    # instantiate model
    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=3)

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
    #cpl = model.calc_cpl(torch.tensor(0.), prd.start, prd.delta, prd.strike, notional)
    #print('Semi-analytic Price =', cpl)


    if strike_plot:
        strikes = torch.linspace(0.025, 0.14, 10)
        times = torch.tensor([1., 2., 5.])
        mc_prices = torch.empty(len(strikes) * len(times))
        i = 0

        for t, s in itertools.product(times, strikes):
            prd = Caplet(
                strike=s,
                start=t,
                delta=delta,
                notional=notional
            )
            dTL = torch.linspace(0.0, float(t + delta), int(50 * (t + delta) + 1))
            cashflows = mcSim(prd, model, rng, N, dTL)
            payoff = torch.sum(cashflows, dim=0)
            # mc
            mc_price = torch.nanmean(payoff)
            mc_prices[i] = mc_price

            i += 1


        mcprices1 = mc_prices[:10]
        mcprices2 = mc_prices[10:20]
        mcprices3 = mc_prices[20:31]

        plt.figure()
        plt.plot(strikes, mcprices1, color='orange', label=r'$1Y$ MC', linestyle='--')
        plt.plot(strikes, mcprices2, color='blue', label=r'$2Y$ MC', linestyle='--')
        plt.plot(strikes, mcprices3, color='green', label=r'$5Y$ MC', linestyle='--')
        plt.xlabel('K')
        plt.ylabel('Price')

        plt.legend(title='Reset date', loc='upper right', fancybox=True)
        plt.title(f'3M Caplet Monte-Carlo N = {notional}')
        if save_fig:
            plt.savefig(get_plot_path('trolle_schwartz/ts_3D_cpl_strikes.png'), dpi=400)
        plt.show()
