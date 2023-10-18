import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import Caplet
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.regressor import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 4096
    N = 1024
    M = 100  # Hedge points

    # Setup Differential Regressor, and Scalar
    deg = 5
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.08)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = mdl.calc_swap_rate(r0, exerciseDate, delta)

    prd = Caplet(
        strike=strike,
        start=exerciseDate,
        delta=delta
    )

    # Simulate paths
    dTL = torch.linspace(0.0, float(exerciseDate + delta), M + 1)
    rng = RNG(seed=seed, use_av=True)
    mcSimPaths(prd, mdl, rng, N, dTL)
    r = mdl.x

    # Find index of the exercise date
    last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])

    """ Estimate Delta using Differential Regression """

    # Make helper functions
    def calc_dfwd_dr(x, s):
        """
        :param  x:    Short rate r0
        :param  s:    Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """
        def _cpl(x):
            fwd = mdl.calc_fwd(x, exerciseDate - s, delta)[0]
            return fwd
        ones = torch.ones_like(x)
        res = jvp(_cpl, x, ones, create_graph=False)
        return res

    def calc_dcpl_dr(x, s):
        """
        :param  x:    Short rate r0
        :param  s:    Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        def _payoffs(x):
            cMdl = Vasicek(a, b, sigma, x, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = Caplet(
                strike=strike,
                start=exerciseDate - s,
                delta=delta
            )
            payoffs = mcSim(cPrd, cMdl, rng, N_train)
            return payoffs

        ones = torch.ones_like(x)
        res = jvp(_payoffs, x, ones, create_graph=False)
        return res

    """ Delta Hedge Experiment """

    def calc_delta(fwd, s):
        # Generate training data
        r0_grid = torch.linspace(0.03, 0.15, N_train)
        F, dFdr = calc_dfwd_dr(r0_grid, s)
        y, dydr = calc_dcpl_dr(r0_grid, s)

        X_train = F.reshape(-1, 1)
        y_train = y.reshape(-1, 1)
        z_train = (dydr / dFdr).reshape(-1, 1)

        # Standardize training data
        X_train, y_train, z_train = scalar.fit_transform(X_train, y_train, z_train)

        # Fit / train estimator
        diff_reg.fit(X_train, y_train, z_train)

        # Predict payoffs and deltas
        fwd = fwd.reshape(-1, 1)
        fwd_trans, _, _ = scalar.transform(fwd)
        y_pred, z_pred = diff_reg.predict(fwd_trans, predict_derivs=True)

        # Transform (un-standardize) predictions
        _, y_pred, z_pred = scalar.predict(None, y_pred, z_pred)

        return z_pred.flatten()

    # Get price of claim (no need to simulate as we have an analytical expression)
    cpl = mdl.calc_cpl(r0, exerciseDate, delta, strike)[0]

    # Initialize experiment
    fwd = mdl.calc_fwd(r[0, :], exerciseDate, delta)[0]

    V = cpl * torch.ones_like(r[0, :])
    h_a = calc_delta(fwd, 0.0)
    h_b = V - h_a * fwd

    r_grid = torch.linspace(0.03, 0.15, 101)
    fwd_grid = mdl.calc_fwd(r_grid, exerciseDate, delta)[0]
    delta_grid = calc_delta(fwd_grid, 0.0)

    # Loop over time
    for k in range(1, last_idx + 1):
        dt = dTL[k] - dTL[k-1]
        s = dTL[k]

        # Update market variables
        fwd = mdl.calc_fwd(r[k, :], exerciseDate - s, delta)[0]

        # Update portfolio
        V = h_a * fwd + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        if k < last_idx:
            h_a = calc_delta(fwd, s)
            h_b = V - h_a * fwd
            print(h_a.min(), h_a.max())

    plt.figure()
    plt.plot(fwd, delta * max0(fwd - strike), 'o', color='blue', label='Payoff function')
    plt.plot(fwd, V, 'o', color='red', label='Value of Hedge Portfolio')
    plt.xlabel('Fwd(T)')
    plt.legend()
    plt.show()
