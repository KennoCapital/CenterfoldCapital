import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import EuropeanPayerSwaption
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.regressor import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSim, RNG

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 4096

    # Setup Differential Regressor, and Scaler
    deg = 5
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=True)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    swapFirstFixingDate = exerciseDate
    swapLastFixingDate = exerciseDate + torch.tensor(5.0)
    notional = torch.tensor(1e6)

    t_swap_fixings = torch.linspace(
        float(swapFirstFixingDate),
        float(swapLastFixingDate),
        int((swapLastFixingDate - swapFirstFixingDate) / delta + 1)
    )

    strike = mdl.calc_swap_rate(r0, t_swap_fixings, delta)

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    """ Estimate Delta using Differential Regression """

    # Make helper functions
    def calc_dswap_dr(x, s):
        """
        :param  x:    Short rate r0
        :param  s:    Current time

        returns:
            tuple with: (Swap Prices, Swap Prices differentiated wrt. r0 evaluated at x)
        """
        def _swap_price(x):
            tau = t_swap_fixings - s
            S = mdl.calc_swap(x, tau, delta, strike, notional)
            return S
        ones = torch.ones_like(x)
        res = jvp(_swap_price, x, ones, create_graph=False)
        return res

    def calc_dswpt_dr(x, s):
        """
        :param  x:    Short rate r0
        :param  s:    Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        def _payoffs(x):
            cMdl = Vasicek(a, b, sigma, x, use_ATS=True, use_euler=False, measure=measure)
            cPrd = EuropeanPayerSwaption(
                    strike=strike,
                    exerciseDate=exerciseDate - s,
                    delta=delta,
                    swapFirstFixingDate=swapFirstFixingDate - s,
                    swapLastFixingDate=swapLastFixingDate - s,
                    notional=notional
            )
            payoffs = mcSim(cPrd, cMdl, rng, N_train)
            return payoffs

        ones = torch.ones_like(x)
        res = jvp(_payoffs, x, ones, create_graph=False)
        return res

    """ Plot MC swaption price against r0 and swap(0) """
    r_grid = torch.linspace(0.03, 0.15, 101)
    swpt_grid = torch.full_like(r_grid, torch.nan)
    for j in range(len(r_grid)):
        tmp_mdl = Vasicek(a, b, sigma, r_grid[j], use_ATS=True, use_euler=False, measure='terminal')
        tmp_rng = RNG(seed=seed, use_av=True)
        swpt_grid[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 50000)))

    swap_grid = tmp_mdl.calc_swap(r_grid, t_swap_fixings, delta, strike, notional)
    dswpt_dswap = swpt_grid.diff() / swap_grid.diff()

    plt.figure()
    plt.plot(r_grid, swpt_grid)
    plt.ylabel('Swpt Price')
    plt.xlabel('r0')
    plt.show()

    plt.figure()
    plt.plot(swap_grid, swpt_grid)
    plt.ylabel('Swpt Price')
    plt.xlabel('Swap(0)')
    plt.show()

    plt.figure()
    plt.plot(swap_grid[1:], dswpt_dswap)
    plt.title('Swpt Delta (Bump and Reval)')
    plt.ylabel('Delta')
    plt.xlabel('Swap(0)')
    plt.show()

    """ Plot Differential Regression (in sample) """

    r0_grid = torch.linspace(0.03, 0.15, N_train)

    swap, dSdr = calc_dswap_dr(r0_grid, 0.0)
    y, dydr = calc_dswpt_dr(r0_grid, 0.0)

    X_train = swap.reshape(-1, 1)
    y_train = y.reshape(-1, 1)
    z_train = (dydr / dSdr).reshape(-1, 1)

    X_train, y_train, z_train = scalar.fit_transform(X_train, y_train, z_train)

    diff_reg.fit(X_train, y_train, z_train)
    y_pred, z_pred = diff_reg.predict(X_train, predict_derivs=True)  # Here X_train is what makes the plot 'in-sample'

    _, y_pred, z_pred = scalar.predict(None, y_pred, z_pred)

    plt.figure()
    plt.plot(swap, y, 'o', color='gray', alpha=0.25, label='Sample Payoffs')
    plt.plot(swap, y_pred, label='Predictions', color='orange')
    plt.title('Learning Payoffs')
    plt.xlabel(r0)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(swap, dydr / dSdr, 'o', color='gray', alpha=0.25, label='Sample Differentials')
    plt.plot(swap, z_pred, label='Predictions', color='orange')
    plt.plot(swap_grid[1:], dswpt_dswap, color='black', label='bump and reval')
    plt.xlabel('Swap(0)')
    plt.title('Learning Sensitivities')
    plt.legend()
    plt.show()
