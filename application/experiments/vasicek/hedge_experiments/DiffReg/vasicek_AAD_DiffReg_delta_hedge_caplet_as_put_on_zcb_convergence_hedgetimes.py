import torch
from torch.autograd.functional import jvp
from tqdm import tqdm
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import CapletAsPutOnZCB
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import diff_reg_fit_predict, log_plotter_without_conv

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

    r0_min = 0.00
    r0_max = 0.16

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Setup Differential Regressor, and Scalar
    deg = 7
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
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = mdl.calc_swap_rate(r0.median(), exerciseDate, delta)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )


    """ Helper functions for calculating pathwise payoffs and deltas, and generating training data """
    def calc_dzcb_dr(r0_vec, t0):
        """
        :param  r0_vec:    Current Short rate r0
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """

        def _zcb(r0_vec):
            zcb = mdl.calc_zcb(r0_vec, exerciseDate - t0 + delta)[0]
            return zcb

        ones = torch.ones_like(r0_vec)
        res = jvp(_zcb, r0_vec, ones, create_graph=False)
        return res


    def calc_dcpl_dr(r0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """

        def _payoffs(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL > exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec))
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res


    """ Delta Hedge Experiment """
    # Simulate paths
    hedge_times = [10, 25, 50, 100, 250, 500, 1000]
    hedge_error = []

    for steps in tqdm(hedge_times):
        dTL = torch.linspace(0.0, float(exerciseDate), steps + 1)
        mcSimPaths(prd, mdl, rng, N_test, dTL)
        r = mdl.x

        # Get price of claim (no need to simulate as we have an analytical expression)
        cpl = torch.empty_like(r[0, :])
        for n in range(N_test):
            mdl.r0 = r[0, n]
            cpl[n] = mdl.calc_cpl(r[0, n], exerciseDate, delta, strike, notional)[0]

        # Initialize experiment
        B = torch.ones((N_test,))
        zcb = mdl.calc_zcb(r[0, :], exerciseDate + delta)[0]

        V = cpl
        h_a = diff_reg_fit_predict(u_vec=zcb, t0=0.0, r0_vec=r0_vec,
                                   calc_dU_dr=calc_dzcb_dr, calc_dPrd_dr=calc_dcpl_dr,
                                   diff_reg=diff_reg, use_av=use_av)[1].flatten()
        h_b = (V - h_a * zcb) / B

        # Loop over time
        for k in range(1, len(dTL)):
            dt = dTL[k] - dTL[k - 1]
            t = dTL[k]

            # Update market variables
            zcb = mdl.calc_zcb(r[k, :], exerciseDate - t + delta)[0]

            # Update portfolio
            B *= torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
            V = h_a * zcb + h_b * B

            if k < len(dTL) - 1:
                r0_vec = choose_training_grid(r[k, :], N_train)
                h_a = diff_reg_fit_predict(u_vec=zcb, r0_vec=r0_vec, t0=t,
                                           calc_dU_dr=calc_dzcb_dr, calc_dPrd_dr=calc_dcpl_dr,
                                           diff_reg=diff_reg, use_av=use_av)[1].flatten()
                h_b = (V - h_a * zcb) / B

        zcbT = mdl.calc_zcb(r[-1, :], delta)[0]
        df = mdl.calc_zcb(r[-1, :], delta)[0]
        K_bar = 1.0 + delta * strike
        payoff_func = notional * K_bar * max0(1.0 / K_bar - zcbT) * df

        hedge_error.append(torch.std(V - payoff_func))

    """ Plot """
    log_plotter_without_conv(X=hedge_times,
                Y=hedge_error,
                title_add=prd.name + f'alpha = {alpha}, Deg={deg}, {N_train} Samples, Notional = {notional}',
                save=False,
                file_name='vasicek_AAD_DiffReg_delta_hedge_caplet_as_put_on_zcb_convergence_hedgetimes',
                xLabel='Hedge Frequency',
                yLabel='Std. of Hedge Error')


