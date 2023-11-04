import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
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

    hedge_times = 100

    r0_min = 0.02
    r0_max = 0.12

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

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

    strike = mdl.calc_swap_rate(r0.median(), t_swap_fixings, delta)

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    # Simulate paths
    dTL = torch.linspace(0.0, float(swapFirstFixingDate), hedge_times + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

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

    # Get price of claim (we use 500k simulations to get an accurate estimate)
    swpt = torch.empty_like(r[0, :])
    for n in range(N_test):
        mdl.r0 = r[0, n]
        swpt[n] = torch.mean(mcSim(prd, mdl, rng, 500000))

    # Initialize experiment
    swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

    V = swpt
    h_a = calc_delta_diff_reg(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                              calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr, diff_reg=diff_reg, use_av=use_av)
    h_b = V - h_a * swap

    # Loop over time
    for k in tqdm(range(1, len(dTL))):
        dt = dTL[k] - dTL[k-1]
        t = dTL[k]

        # Update market variables
        swap = mdl.calc_swap(r[k, :], t_swap_fixings - t, delta, strike, notional)

        # Update portfolio
        V = h_a * swap + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        if k < len(dTL) - 1:
            r0_vec = choose_training_grid(r[k, :], N_train)
            h_a = calc_delta_diff_reg(u_vec=swap, r0_vec=r0_vec, t0=t,
                                      calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr, diff_reg=diff_reg, use_av=use_av)
            h_b = V - h_a * swap

    swapT = torch.linspace(float(swap.min()), float(swap.max()), 1001)
    payoff_func = max0(swapT)

    MAE_value = torch.mean(torch.abs(V - max0(swap)))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots(1)
    ax.plot(swapT, payoff_func, color='black', label='Payoff function')
    ax.plot(swap, V, 'o', color='orange', label='Value of Hedge Portfolio', alpha=0.5)
    ax.set_xlabel('Swap(T)')
    ax.text(0.05, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax.transAxes)

    # Adjust size of plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    fig.suptitle(prd.name + f'\nHedgeFreq={dTL[1]:.4g}, alpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str)

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    #plt.savefig(get_plot_path('vasicek_AAD_DiffReg_delta_hedge_EuPayerSwpt.png'), dpi=400)
    plt.show()

