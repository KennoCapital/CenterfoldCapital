from application.engine.products import Caplet
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.vasicek import Vasicek
import torch
from application.engine.mcBase import mcSim, RNG
import matplotlib.pyplot as plt

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    # Setup
    seed = 1234
    N = 1024*10
    measure = 'risk_neutral'
    produce_plots = True
    perform_calibration = False
    gradient_plot = True

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553) #0553
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045) #045
    alpha1 = torch.tensor(0.131) #131
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542) #7542
    #
    varphi = torch.tensor(0.0832)

    # Product specification
    start = torch.tensor(20.0)
    delta = torch.tensor(.25)
    strike = torch.tensor(0.084)
    notional = torch.tensor(1.0)

    dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

    # instantiate model
    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi)

    rng = RNG(seed=seed, use_av=False)

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
    cpl = model.calc_cpl(0, prd.start, prd.delta, prd.strike, notional)
    print('Semi-analytic Price =', cpl)

    import scipy

    def calibrate_trolle_schwartz_cpl_price(maturities, strikes, market_prices, alpha0=0.045, alpha1=0.131):
        maturities = torch.tensor(maturities)

        def obj(alpha):
            alpha = torch.tensor(alpha)
            alpha0, alpha1 = alpha[0], alpha[1]
            model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1)
            model_prices = torch.empty_like(market_prices, dtype=torch.float64)

            rng = RNG(seed=seed, use_av=False)

            for i,T in enumerate(maturities):

                prd = Caplet(
                    strike=strikes[i],
                    start=T,
                    delta=delta,
                    notional=notional
                )
                cashflows = mcSim(prd, model, rng, N, dTL)
                payoff = torch.sum(cashflows, dim=0)
                mc_price = torch.nanmean(payoff)
                model_prices[i] = mc_price

            err = model_prices - market_prices
            mse = torch.linalg.norm(err) ** 2
            return mse

        return scipy.optimize.minimize(
                fun=obj, x0=torch.tensor([alpha0, alpha1]), method='Nelder-Mead', tol=1e-12,
                bounds=[(0.0001, 2.000), (0.0001, 2.000)],
                options={
                    'xatol': 1e-12,
                    'fatol': 1e-12,
                    'maxiter': 2500,
                    'maxfev': 2500,
                    'adaptive': True,
                    'disp': True
                })


    maturities = [0.25, 1.0, 5.0, 10.0, 15.0, 20.0]
    strikes = torch.empty_like(torch.tensor(maturities), dtype=torch.float64)

    if perform_calibration:

        market_prices = torch.empty_like(strikes, dtype=torch.float64)
        model_prices = torch.empty_like(market_prices, dtype=torch.float64)
        delta = torch.tensor(0.25)

        # Market prices (i.e. Vasicek prices)
        a_ = torch.tensor(0.86)
        b_ = torch.tensor(0.09)
        sigma_ = torch.tensor(0.0148)
        r0_ = torch.tensor(0.08)

        for i, T in enumerate(maturities):
            market_model = Vasicek(a_, b_, sigma_)
            swap_rate = market_model.calc_swap_rate(r0_, torch.tensor(T), delta)
            cpl = market_model.calc_cpl(r0_, torch.tensor(T), delta, torch.tensor(0.084), notional)[0][0]
            strikes[i] = torch.tensor(0.084)
            market_prices[i] = cpl

        # Calibration

        calib = calibrate_trolle_schwartz_cpl_price(maturities, strikes, market_prices)
        alpha0_cal, alpha1_cal = torch.tensor(calib.x)

        # Model prices
        model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1)
        for i, T in enumerate(maturities):
            prd = Caplet(
                strike=strikes[i],
                start=torch.tensor(T),
                delta=delta,
                notional=notional
            )
            cashflows = mcSim(prd, model, rng, N, dTL)
            payoff = torch.sum(cashflows, dim=0)
            mc_price = torch.nanmean(payoff)
            model_prices[i] = mc_price

        # Error
        err = model_prices - market_prices
        print('Model price - Market price = {}'.format(err))
        print('MSE = {}'.format(float(torch.sum(err**2))))
        print('alpha0 = {}, alpha1 = {}'.format(
            calib.x[0].round(4), calib.x[1].round(4)
        ))



    """
    x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i for i in model.x]
    r0 = model.calc_short_rate(
        [x[:, 0, :], v[:, 0, :], phi1[:, 0, :], phi2[:, 0, :], phi3[:, 0, :], phi4[:, 0, :], phi5[:, 0, :],
         phi6[:, 0, :]], t=0.0)

    f00 = model.calc_instant_fwd( [x[:, 0, :], v[:, 0, :], phi1[:, 0, :], phi2[:, 0, :], phi3[:, 0, :], phi4[:, 0, :], phi5[:, 0, :], phi6[:, 0, :]], t=0.0, T=0.0)
    f0T = model.calc_instant_fwd( [x[:, -1, :], v[:, -1, :], phi1[:, -1, :], phi2[:, -1, :], phi3[:, -1, :], phi4[:, -1, :], phi5[:, -1, :], phi6[:, -1, :]], t=0.0, T=start)

    # zcb price time-0 using trapezoidal rule
    zcb0 = torch.exp(-0.5 * (f00 + f0T) * start)
    print('zcb0', zcb0.mean())
    """
    maturities = [0.25, 1.0, 5.0, 10.0, 15.0, 20.0]
    strikes.fill_(torch.tensor(0.084))
    state_vars = torch.concat(model.x)
    zcb_term = torch.zeros_like(strikes)
    for i,T in enumerate(maturities):
        zcb_term[i] = model.calc_zcb_price(state_vars[:,1,:], torch.tensor(0.), torch.tensor(T)).mean()

    plt.figure()
    plt.plot(maturities, zcb_term)
    plt.show()

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
        plt.plot(x[-1][:,0], color='blue', label='x')
        plt.plot(v[-1][:,0], color = 'red', label='v')
        plt.legend()
        plt.title('state vars x & v')
        plt.show()

        plt.figure()
        plt.plot(phi1[-1][:,0], color='blue', label='phi1')
        plt.plot(phi2[-1][:,0], color = 'red', label='phi2')
        plt.plot(phi3[-1][:,0], color = 'green', label='phi3')
        plt.plot(phi4[-1][:,0], color = 'orange', label='phi4')
        plt.plot(phi5[-1][:,0], color = 'purple', label='phi5')
        plt.plot(phi6[-1][:,0], color = 'brown', label='phi6')
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

    if gradient_plot:



        # Define the function for dynamics of N

        def Bx(tau):
            tau = torch.tensor(tau)
            Bx = alpha1 / gamma * (
                    (1 / gamma + alpha0 / alpha1) * (torch.exp(-gamma * tau) - 1) + \
                    tau * torch.exp(-gamma * tau))
            return Bx
        def _compute_dN(N, t):
            u = 8000
            T0=prd.start
            T1 = prd.start + prd.delta
            """ dynamics of N """
            dNdt = N * (-kappa + sigma * rho * (u * Bx(T1 - T0 + t) + (1 - u) * Bx(t))) + \
                   0.5 * N ** 2 * sigma ** 2 + 0.5 * (u ** 2 - u) * Bx(T1 - T0 + t) ** 2 + 0.5 * (
                               (1 - u) ** 2 - (1 - u)) * Bx(t) ** 2 + \
                   u * (1 - u) * Bx(T1 - T0 + t) * Bx(t)

            return dNdt


        # Create a grid of N and t values
        N_values = torch.linspace(0, 1, 20)
        t_values = torch.linspace(0, 0.1, 20)

        # Calculate gradients at each point on the grid
        X, Y = torch.meshgrid(N_values, t_values)
        U = torch.zeros_like(X)
        V = torch.zeros_like(Y)

        u = torch.tensor(2.)

        for i in range(len(N_values)):
            for j in range(len(t_values)):
                U[i, j] = _compute_dN(N_values[i], t_values[j])
                V[i, j] = 1  # Gradient in the t-direction, assuming unit step size

        # Create a quiver plot
        plt.figure(figsize=(8, 6))
        plt.quiver(X, Y, U, V, scale=20, scale_units='inches', angles='xy', color='b', alpha=0.6)
        plt.xlabel('N')
        plt.ylabel('t')
        plt.title('Gradient Plot of the ODE')
        plt.show()


        def _RK4_vectorized(N0, t, discSteps, T0):
            """
            Vectorized Runge-Kutta fourth-order method
            """
            N = N0
            dTau = torch.linspace(T0, t, discSteps)
            h = dTau[1] - dTau[0]

            for i in range(discSteps):
                k1N = _compute_dN(N, dTau[i])
                k2N = _compute_dN(N + 0.5 * h * k1N, dTau[i] + 0.5 * h)
                k3N = _compute_dN(N + 0.5 * h * k2N, dTau[i] + 0.5 * h)
                k4N = _compute_dN(N + h * k3N, dTau[i] + h)

                N += (h / 6) * (k1N + 2 * k2N + 2 * k3N + k4N)

            return N


        # Example usage

        N0 = torch.tensor([0.0001])  # Initial value of N as a numpy array

        t = 0.0
        discSteps = 1024  # Number of discretization steps
        T0 = 1.0

        N_result = _RK4_vectorized(N0, t, discSteps, T0)
        print("Result N:", N_result)

        from scipy.integrate import solve_ivp
        from scipy.integrate import RK45

        solver = solve_ivp(_compute_dN, t_span=[0.,1.], y0=torch.tensor([0.001]), method='BDF')

        torch.trapz(torch.tensor(solver.y[0]) * gamma * kappa, torch.tensor(solver.t))

        plt.figure()
        plt.plot(solver.t, solver.y[0] )
        plt.xlabel('t')
        plt.ylabel('N')
        plt.show()






