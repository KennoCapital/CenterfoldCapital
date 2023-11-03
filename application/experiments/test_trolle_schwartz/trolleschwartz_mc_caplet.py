from application.engine.products import Caplet
from application.engine.trolleSchwartz import trolleSchwartz
import torch
from application.engine.mcBase import mcSim, RNG

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    # Setup
    seed = 1234
    N = 100
    measure = 'risk_neutral'
    produce_plots = False

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553) #0553
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.0131)
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.09) #7542
    r0 = torch.tensor(0.08)

    #todo: these should be initialized @ 0 in model
    x0 = torch.tensor([.0])
    v0= torch.clone(x0)
    phi1_0 = torch.clone(x0)
    phi2_0 = torch.clone(x0)
    phi3_0 = torch.clone(x0)
    phi4_0 = torch.clone(x0)
    phi5_0 = torch.clone(x0)
    phi6_0 = torch.clone(x0)

    # Product specification
    start = torch.tensor(5.0)
    delta = torch.tensor(1.)
    strike = torch.tensor(0.084)
    notional = torch.tensor(1e6)

    dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

    # instantiate model
    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1,
                           x0, v0, phi1_0, phi2_0, phi3_0, phi4_0, phi5_0, phi6_0)

    rng = RNG(seed=seed, use_av=False)

    prd = Caplet(
        strike=strike,
        start=start,
        delta=delta,
        notional=notional
    )


    cashflows = mcSim(prd, model, rng, N, dTL)
    payoff = torch.sum(cashflows, dim=0)

    mc_price = torch.nanmean(payoff)
    print('MC Price =', mc_price)

    #cpl = model.calc_cpl(0, prd.start, prd.delta, prd.strike)
    #print('Semi-analytic Price =', cpl)


    x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i for i in model.x]
    r0 = model.calc_short_rate(
        [x[:, 0, :], v[:, 0, :], phi1[:, 0, :], phi2[:, 0, :], phi3[:, 0, :], phi4[:, 0, :], phi5[:, 0, :],
         phi6[:, 0, :]], t=0.0)

    f00 = model.calc_instant_fwd( [x[:, 0, :], v[:, 0, :], phi1[:, 0, :], phi2[:, 0, :], phi3[:, 0, :], phi4[:, 0, :], phi5[:, 0, :], phi6[:, 0, :]], t=0.0, T=0.0)
    f0T = model.calc_instant_fwd( [x[:, -1, :], v[:, -1, :], phi1[:, -1, :], phi2[:, -1, :], phi3[:, -1, :], phi4[:, -1, :], phi5[:, -1, :], phi6[:, -1, :]], t=0.0, T=start)

    # zcb price time-0 using trapezoidal rule
    zcb0 = torch.exp(-0.5 * (f00 + f0T) * start)
    print('zcb0', zcb0.mean())

    if produce_plots:
        # plot forward rates
        plt.figure()
        plt.plot(model.paths[1].fwd[0][0], color='blue', label='F')
        plt.plot(cashflows[0], color='red', label='cashflows')
        plt.hlines(y=strike, xmin=0, xmax=len(cashflows[0]), color='green', label='strike')
        plt.legend()
        plt.title('Forward rates // payoff')
        plt.show()

        #plot stat vars
        x,v,phi1,phi2,phi3,phi4,phi5,phi6 = [i for i in model.x]

        plt.figure()
        plt.plot(x[0][:,0], color='blue', label='x')
        plt.plot(v[0][:,0], color = 'red', label='v')
        plt.legend()
        plt.title('state vars x & v')
        plt.show()

        plt.figure()
        plt.plot(phi1[0][:,0], color='blue', label='phi1')
        plt.plot(phi2[0][:,0], color = 'red', label='phi2')
        plt.plot(phi3[0][:,0], color = 'green', label='phi3')
        plt.plot(phi4[0][:,0], color = 'orange', label='phi4')
        plt.plot(phi5[0][:,0], color = 'purple', label='phi5')
        plt.plot(phi6[0][:,0], color = 'brown', label='phi6')
        plt.legend()
        plt.title('state vars phi1-6')
        plt.show()

        # plot f variance
        sigma_fct = model._sigma(model.timeline, start + delta)
        v_sqrt = v[0][:,0:5].sqrt()
        colors = ['b', 'r', 'y', 'g', 'c']
        plt.figure()
        for i in range(5):
            plt.plot(v_sqrt[:,i] * sigma_fct, color=colors[i], label='f vol')
        plt.legend()
        plt.xlabel('time steps')
        plt.title('f volatility')
        plt.show()




