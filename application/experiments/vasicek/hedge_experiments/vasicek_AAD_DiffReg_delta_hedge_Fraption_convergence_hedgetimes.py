import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import Fraption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.path_config import get_plot_path
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import calc_delta_diff_reg

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

    hedge_points = 100

    r0_min = 0.07
    r0_max = 0.09

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Setup Differential Regressor, and Scalar
    deg = 15
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(0.25)
    start = torch.tensor(1.0)
    delta = torch.tensor(5.0)
    notional = torch.tensor(1e6)

    strike = mdl.calc_fwd(r0, start, delta)

    prd = Fraption(
        exerciseDate=exerciseDate,
        strike=strike,
        start=start,
        delta=delta,
        notional=notional
    )

""" Helper functions for generating training data of pathwise payoffs and deltas """


    def calc_dFRA_dr(r0_vec, t0):
        """
        :param  r0_vec:    Current Short rate r0
        :param  t0:        Current time

        returns:
            tuple with: (FRA prices, FRA Prices differentiated wrt. r0 evaluated at r0_vec)
        """

        def _FRA(r0_vec):
            return mdl.calc_fra(r0_vec, start - t0, delta, strike, notional)[0]

        ones = torch.ones_like(r0_vec)
        res = jvp(_FRA, r0_vec, ones, create_graph=False)
        return res


    def calc_dFraption_dr(r0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """

        def _payoffs(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = Fraption(
                exerciseDate=exerciseDate - t0,
                strike=strike,
                start=start - t0,
                delta=delta,
                notional=notional
            )
            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec))
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res

    """ Delta Hedge Experiment """


    # Simulate paths
    hedge_times = [10, 25, 50, 100, 250, 500, 1000]
    hedge_error = []

    for steps in hedge_times:

        dTL = torch.linspace(0.0, float(exerciseDate), steps + 1)
        mcSimPaths(prd, mdl, rng, N_test, dTL)
        r = mdl.x

        # Get price of claim (no need to simulate as we have an analytical expression)
        mdl_pricer = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure='risk_neutral')
        fraption = torch.mean(mcSim(prd, mdl, rng, 500000))

        # Initialize experiment
        fra = mdl.calc_fra(r[0, :], start, delta, strike, notional)[0]

        V = fraption * torch.ones_like(r[0, :])
        h_a = calc_delta_diff_reg(u_vec=fra, r0_vec=r0_vec, t0=0.0,
                                  calc_dPrd_dr=calc_dFraption_dr, calc_dU_dr=calc_dFRA_dr, diff_reg=diff_reg,
                                  use_av=use_av)
        h_b = V - h_a * fra

        # Loop over time
        for k in range(1, len(dTL)):
            dt = dTL[k] - dTL[k-1]
            t = dTL[k]

            # Update market variables
            fra = mdl.calc_fra(r[k, :], start - t, delta, strike, notional)[0]

            # Update portfolio
            V = h_a * fra + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
            if k < len(dTL) - 1:
                r0_vec = choose_training_grid(r[k, :], N_train)
                h_a = calc_delta_diff_reg(u_vec=fra, r0_vec=r0_vec, t0=t,
                                          calc_dPrd_dr=calc_dFraption_dr, calc_dU_dr=calc_dFRA_dr, diff_reg=diff_reg,
                                          use_av=use_av)
                h_b = V - h_a * fra

        hedge_error.append(torch.std(V - max0(fra)))

    """ Plot """
    # add convergence order line
    x = np.log(hedge_times)
    y = np.log(hedge_error)
    res = linregress(x, y)
    fit_y_log = res.slope * x + res.intercept

    plt.figure()
    plt.suptitle(prd.name + f'alpha = {alpha}, deg={deg}, {N_train} samples, notional = {notional}')
    plt.title(f'convergence order = {res.slope:.2f}')
    plt.plot(x, fit_y_log, '--', color='red')
    plt.plot(x, y, 'o-', color='blue')

    plt.xlabel('steps per fixing')
    plt.ylabel('std. dev. of hedge error')

    plt.xticks(ticks=x, labels=hedge_times)
    plt.yticks(ticks=y, labels=np.round(y, 2))

    #plt.savefig(get_plot_path('vasicek_AAD_DiffReg_delta_hedge_Fraption_convergence_hedgetimes.png'), dpi=400)
    plt.show()

