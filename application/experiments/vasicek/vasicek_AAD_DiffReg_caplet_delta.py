import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import Caplet
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSim, RNG

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 4096

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

    rng = RNG(seed=seed, use_av=True)

    # Product specification
    exerciseDate = torch.tensor(0.25)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = mdl.calc_swap_rate(r0, exerciseDate, delta)

    prd = Caplet(
        strike=strike,
        start=exerciseDate,
        delta=delta
    )

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

    """ Plot Analytical Caplet price against r0 and Fwd(0, T, T+delta) """

    r_grid = torch.linspace(0.03, 0.15, 1001)
    cpl_grid = mdl.calc_cpl(r_grid, exerciseDate, delta, strike)[0]

    fwd_grid = mdl.calc_fwd(r_grid, exerciseDate, delta)[0]
    dcpl_dfwd = cpl_grid.diff() / fwd_grid.diff()

    plt.figure()
    plt.plot(r_grid, cpl_grid)
    plt.ylabel('Caplet Price')
    plt.xlabel('r0')
    plt.show()

    plt.figure()
    plt.plot(fwd_grid, cpl_grid)
    plt.ylabel('Caplet Price')
    plt.xlabel('Fwd(0, T, T+delta)')
    plt.show()

    plt.figure()
    plt.plot(cpl_grid[1:], dcpl_dfwd)
    plt.title('Caplet Delta (Bump and Reval)')
    plt.ylabel('Delta')
    plt.xlabel('Fwd(0, T, T+delta)')
    plt.show()

    """ Plot Differential Regression (in sample) """
    r0_grid = torch.linspace(0.03, 0.15, N_train)
    '''
    r_std = torch.sqrt(sigma ** 2 / 2 * a * (1 - torch.exp(-2 * a * exerciseDate)))
    r_mean = r0 * torch.exp(-a * exerciseDate) + b * (1 - torch.exp(-a * exerciseDate))
    one = torch.ones(N_train)
    r0_grid = torch.normal(mean=r_mean * one, std=1.5 * r_std * one)
    r0_grid = torch.sort(r0_grid).values
    '''
    fwd, dFdr = calc_dfwd_dr(r0_grid, 0.0)
    y, dydr = calc_dcpl_dr(r0_grid, 0.0)

    X_train = fwd.reshape(-1, 1)
    y_train = y.reshape(-1, 1)
    z_train = (dydr / dFdr).reshape(-1, 1)

    X_train, y_train, z_train = scalar.fit_transform(X_train, y_train, z_train)

    diff_reg.fit(X_train, y_train, z_train)
    y_pred, z_pred = diff_reg.predict(X_train, predict_derivs=True)  # Here X_train is what makes the plot 'in-sample'

    _, y_pred, z_pred = scalar.predict(None, y_pred, z_pred)

    plt.figure()
    plt.plot(fwd, y, 'o', color='gray', alpha=0.25, label='Sample Payoffs')
    plt.plot(fwd, y_pred, label='DiffReg', color='orange')
    plt.plot(fwd_grid, cpl_grid, color='black', label='bump and reval')
    plt.title('Learning Payoffs')
    plt.xlabel('r0')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(fwd, dydr / dFdr, 'o', color='gray', alpha=0.25, label='Sample Differentials')
    plt.plot(fwd, z_pred, label='DiffReg', color='orange')
    plt.plot(fwd_grid[1:], dcpl_dfwd, color='black', label='bump and reval')
    plt.xlabel('Fwd(0, T, T+delta)')
    plt.title('Learning Sensitivities')
    plt.legend()
    plt.show()
