import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import EuropeanPayerSwaption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0
from application.utils.path_config import get_plot_path
from application.experiments.vasicek.vasicek_hedge_tools import calc_delta_diff_reg

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

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

    """ Helper functions for generating training data of pathwise payoffs and deltas """
    def calc_dswap_dr(r0_vec: torch.Tensor, t0: float):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Swap Prices, Swap Prices differentiated wrt. r0 evaluated at entries in r0_vec)
        """
        def _swap_price(r0_vec):
            tau = t_swap_fixings - t0
            return mdl.calc_swap(r0_vec, tau, delta, strike, notional)

        ones = torch.ones_like(r0_vec)
        res = jvp(_swap_price, r0_vec, ones, create_graph=False)
        return res

    def calc_dswpt_dr(r0_vec: torch.Tensor, t0: float):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at entries in r0_vec)
        """
        def _payoffs(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure=measure)
            cPrd = EuropeanPayerSwaption(
                    strike=strike,
                    exerciseDate=exerciseDate - t0,
                    delta=delta,
                    swapFirstFixingDate=swapFirstFixingDate - t0,
                    swapLastFixingDate=swapLastFixingDate - t0,
                    notional=notional
            )

            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec))
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res

    """ Delta Hedge Experiment """


    # Simulate paths
    steps_per_fixing = [5, 10, 15]
    hedge_error = []

    for steps in steps_per_fixing:

        dTL = torch.linspace(0.0, float(swapLastFixingDate), steps * int(swapLastFixingDate / delta) + 1)
        mcSimPaths(prd, mdl, rng, N_test, dTL)
        r = mdl.x

        # Find index of the exercise date
        last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])


        # Get price of claim (we use 500k simulations to get an accurate estimate)
        swpt = torch.mean(mcSim(prd, mdl, rng, 500000))

        # Initialize experiment
        swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

        V = swpt * torch.ones_like(r[0, :])
        h_a = calc_delta_diff_reg(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                                  calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr, diff_reg=diff_reg, use_av=use_av)
        h_b = V - h_a * swap

        # Loop over time
        for k in range(1, last_idx + 1):
            dt = dTL[k] - dTL[k-1]
            t = dTL[k]

            # Update market variables
            swap = mdl.calc_swap(r[k, :], t_swap_fixings - t, delta, strike, notional)

            # Update portfolio
            V = h_a * swap + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
            if k < last_idx:
                r0_vec = choose_training_grid(r[k, :], N_train)
                h_a = calc_delta_diff_reg(u_vec=swap, r0_vec=r0_vec, t0=t,
                                          calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr, diff_reg=diff_reg, use_av=use_av)
                h_b = V - h_a * swap

        swapT = torch.linspace(float(swap.min()), float(swap.max()), 1001)
        payoff_func = max0(swapT)

        hedge_error.append(torch.std(V - max0(swap)))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots(1)
    ax.plot(steps_per_fixing, hedge_error, 'o', color='black', label='hedge_error')
    ax.set_xlabel('steps per fixing')

    # Adjust size of plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    fig.suptitle(prd.name + f'alpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str)

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    plt.savefig(get_plot_path('vasicek_AAD_DiffReg_Fraption_delta_hedge.png'), dpi=400)
    plt.show()

