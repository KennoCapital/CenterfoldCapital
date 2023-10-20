import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import EuropeanPayerSwaption
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 4096
    N = 1024
    steps_per_fixing = 10

    # Setup Differential Regressor, and Scalar
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

    # Simulate paths
    dTL = torch.linspace(0.0, float(swapLastFixingDate), steps_per_fixing * int(swapLastFixingDate / delta) + 1)
    rng = RNG(seed=seed, use_av=True)
    mcSimPaths(prd, mdl, rng, N, dTL)
    r = mdl.x

    # Find index of the exercise date
    last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])

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
            cMdl = Vasicek(a, b, sigma, x, use_ATS=True, use_euler=False, measure='terminal')
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

    """ Delta Hedge Experiment """

    def calc_delta(swap, s):
        # Generate training data
        r0_grid = torch.linspace(0.03, 0.15, N_train)
        S, dSdr = calc_dswap_dr(r0_grid, s)
        y, dydr = calc_dswpt_dr(r0_grid, s)

        X_train = S.reshape(-1, 1)
        y_train = y.reshape(-1, 1)
        z_train = (dydr / dSdr).reshape(-1, 1)

        # Standardize training data
        X_train, y_train, z_train = scalar.fit_transform(X_train, y_train, z_train)

        # Fit / train estimator
        diff_reg.fit(X_train, y_train, z_train)

        # Predict payoffs and deltas
        swap = swap.reshape(-1, 1)

        swap_trans, _, _ = scalar.transform(swap)
        y_pred, z_pred = diff_reg.predict(swap_trans, predict_derivs=True)

        # Transform (un-standardize) predictions
        _, y_pred, z_pred = scalar.predict(None, y_pred, z_pred)

        return z_pred.flatten()

    # Get price of claim (we use 500k simulations to get an accurate estimate)
    swpt = torch.mean(mcSim(prd, mdl, rng, 500000))

    # Initialize experiment
    swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

    V = swpt * torch.ones_like(r[0, :])
    h_a = calc_delta(swap, 0.0)
    h_b = V - h_a * swap

    # Loop over time
    for k in range(1, last_idx + 1):
        dt = dTL[k] - dTL[k-1]
        s = dTL[k]

        # Update market variables
        swap = mdl.calc_swap(r[k, :], t_swap_fixings - s, delta, strike, notional)

        # Update portfolio
        V = h_a * swap + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        if k < last_idx:
            h_a = calc_delta(swap, s)
            h_b = V - h_a * swap

    plt.figure()
    plt.plot(swap, max0(swap), 'o', color='blue', label='Payoff function')
    plt.plot(swap, V, 'o', color='red', label='Value of Hedge Portfolio')
    plt.xlabel('Swap(T)')
    plt.legend()
    plt.show()
