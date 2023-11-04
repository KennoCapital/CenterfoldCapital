import torch
from tqdm import tqdm
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import Fraption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import calc_delta_diff_reg, log_plotter

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    seed = 1234
    N_test = 256
    use_av = True

    r0_min = 0.02
    r0_max = 0.12

    # Setup Differential Regressor, and Scalar
    deg = 15
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)

    # Model specification
    r0 = torch.linspace(r0_min, r0_max, N_test)
    a = torch.tensor(0.86)
    b = r0.median()
    sigma = torch.tensor(0.0148)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(0.25)
    start = torch.tensor(1.0)
    delta = torch.tensor(5.0)
    notional = torch.tensor(1e6)

    strike = mdl.calc_fwd(r0.median(), start, delta)

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
    steps = 100
    dTL = torch.linspace(0.0, float(exerciseDate), steps + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    # Simulate paths
    N_train = [256 * i**2 for i in range(1, 10)]
    hedge_error = []

    for N in tqdm(N_train):

        r0_vec = torch.linspace(r0_min, r0_max, N)

        # Get price of claim (no need to simulate as we have an analytical expression)
        fraption = torch.empty_like(r[0, :])
        for n in range(N_test):
            mdl.r0 = r[0, n]
            fraption[n] = torch.mean(mcSim(prd, mdl, rng, 500000))

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
                r0_vec = choose_training_grid(r[k, :], N)
                h_a = calc_delta_diff_reg(u_vec=fra, r0_vec=r0_vec, t0=t,
                                          calc_dPrd_dr=calc_dFraption_dr, calc_dU_dr=calc_dFRA_dr, diff_reg=diff_reg,
                                          use_av=use_av)
                h_b = V - h_a * fra

        hedge_error.append(torch.std(V - max0(fra)))

    """ Plot """
    log_plotter(X=N_train,
                Y=hedge_error,
                title_add=prd.name + f'alpha = {alpha}, deg={deg}, times hedging = {steps}, notional = {notional}',
                save=False,
                file_name='vasicek_AAD_DiffReg_delta_hedge_Fraption_convergence_trainingsamples')

