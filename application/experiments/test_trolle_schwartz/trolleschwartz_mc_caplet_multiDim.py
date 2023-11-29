from application.engine.products import Caplet
from application.engine.trolleSchwartz import trolleSchwartz
import torch
from application.engine.mcBase import mcSim, RNG
import matplotlib.pyplot as plt
from tqdm.contrib import itertools


torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":

    plot_state_vars = True
    strike_plot = False
    euler_step_size = 500

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
    #
    varphi = torch.tensor(0.0668)

    # Product specification
    start = torch.tensor(1.)
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
        plt.plot(strikes, mcprices1, label='1')
        plt.plot(strikes, mcprices2, label='2.5')
        plt.plot(strikes, mcprices3, label='5')
        plt.xlabel('strike')
        plt.ylabel('price')
        plt.legend()
        plt.title('TS cpl MC price' + f' N = {notional} with delta = {delta}')
        plt.show()


    if plot_state_vars:
        """ Plotting state variables """
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i.squeeze()[:, :, 420] for i in model.x]

        fig1, axs1 = plt.subplots(1, 2, figsize=(8, 4))
        axs1[0].plot(dTL, x[0,:], label=r'$x_1$', linestyle='-', color='black')
        axs1[0].plot(dTL, x[1, :], label=r'$x_2$', linestyle='-.', color='black')
        axs1[0].plot(dTL, x[2, :], label=r'$x_3$', linestyle=':', color='black')
        axs1[0].legend(loc='upper left')
        axs1[0].set_xlabel('Years')
        axs1[1].plot(dTL, v[0,:], label=r'$\nu_1$', linestyle='-',color='black')
        axs1[1].plot(dTL, v[1, :], label=r'$\nu_2$', linestyle='-.', color='black')
        axs1[1].plot(dTL, v[2, :], label=r'$\nu_3$', linestyle=':', color='black')
        axs1[1].legend(loc='upper left')
        axs1[1].set_xlabel('Years')


        fig2, axs2 = plt.subplots(3, 2, figsize=(7, 8))
        axs2[0, 0].plot(dTL, phi1[0, :], label=r'$\phi_{1,1}$', linestyle='-', color='black')
        axs2[0, 0].plot(dTL, phi1[1, :], label=r'$\phi_{1,2}$', linestyle='-.', color='black')
        axs2[0, 0].plot(dTL, phi1[2, :], label=r'$\phi_{1,3}$', linestyle=':', color='black')
        axs2[0, 0].legend(loc='upper left')
        axs2[0, 1].plot(dTL, phi2[0, :], label=r'$\phi_{2,1}$', linestyle='-', color='black')
        axs2[0, 1].plot(dTL, phi2[1, :], label=r'$\phi_{2,2}$', linestyle='-.', color='black')
        axs2[0, 1].plot(dTL, phi2[2, :], label=r'$\phi_{2,3}$', linestyle=':', color='black')
        axs2[0, 1].legend(loc='upper left')
        axs2[1, 0].plot(dTL, phi3[0, :], label=r'$\phi_{3,1}$', linestyle='-', color='black')
        axs2[1, 0].plot(dTL, phi3[1, :], label=r'$\phi_{3,2}$', linestyle='-.', color='black')
        axs2[1, 0].plot(dTL, phi3[2, :], label=r'$\phi_{3,3}$', linestyle=':', color='black')
        axs2[1, 0].legend(loc='upper left')
        axs2[1, 1].plot(dTL, phi4[0, :], label=r'$\phi_{4,1}$', linestyle='-', color='black')
        axs2[1, 1].plot(dTL, phi4[1, :], label=r'$\phi_{4,2}$', linestyle='-.', color='black')
        axs2[1, 1].plot(dTL, phi4[2, :], label=r'$\phi_{4,3}$', linestyle=':', color='black')
        axs2[1, 1].legend(loc='upper left')
        axs2[2, 0].plot(dTL, phi5[0, :], label=r'$\phi_{5,1}$', linestyle='-', color='black')
        axs2[2, 0].plot(dTL, phi5[1, :], label=r'$\phi_{5,2}$', linestyle='-.', color='black')
        axs2[2, 0].plot(dTL, phi5[2, :], label=r'$\phi_{5,3}$', linestyle=':', color='black')
        axs2[2, 0].legend(loc='upper left')
        axs2[2, 0].set_xlabel('Years')
        axs2[2, 1].plot(dTL, phi6[0, :], label=r'$\phi_{6,1}$', linestyle='-', color='black')
        axs2[2, 1].plot(dTL, phi6[1, :], label=r'$\phi_{6,2}$', linestyle='-.', color='black')
        axs2[2, 1].plot(dTL, phi6[2, :], label=r'$\phi_{6,3}$', linestyle=':', color='black')
        axs2[2, 1].legend(loc='upper left')
        axs2[2, 1].set_xlabel('Years')

        fig1.tight_layout()
        fig2.tight_layout()
        plt.show()

        # plotting forward rate vol (hump shaped!)
        fwdVol = model.fwd_rate_vol(0, dTL)
        plt.figure()
        plt.plot(dTL, fwdVol[0], label=r'$\sigma_{f,1}(\tau)$', linestyle='-', color='C0')
        plt.plot(dTL, fwdVol[1], label=r'$\sigma_{f,2}(\tau)$', linestyle='-.', color='C0')
        plt.plot(dTL, fwdVol[2], label=r'$\sigma_{f,3}(\tau)$', linestyle=':', color='C0')
        plt.legend()
        plt.xlabel('Years')
        plt.ylim(0, None)
        plt.xlim(0, None)
        plt.show()

        # plotting forward rate movements
        fwds = torch.empty(len(model.timeline))
        for t in range(len(model.timeline)):
            x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i.squeeze()[:, t, :] for i in model.x]
            state = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
            fwds[t] = model.calc_fwd(state, start, delta).mean(dim=1)

        plt.figure()
        plt.plot(model.timeline, fwds)
        plt.show()
