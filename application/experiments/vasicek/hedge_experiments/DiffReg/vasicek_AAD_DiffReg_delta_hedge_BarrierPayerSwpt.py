import torch
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import BarrierPayerSwaption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0
from application.utils.path_config import get_plot_path, get_data_path
from application.experiments.vasicek.vasicek_hedge_tools import diff_reg_fit_predict

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    # Price data is generated without smoothing
    file_path = get_data_path('vasicek_Barrier_test_set_hedge.pkl')

    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

    r0_min = -0.02
    r0_max = 0.15

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=True)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.linspace(r0_min, r0_max, N_test) #torch.tensor(0.08)
    measure = 'terminal'

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

    strike = torch.tensor(0.0871)  # mdl.calc_swap_rate(r0, t_swap_fixings, delta)

    hedge_times = 100
    dTL = torch.linspace(0.0, float(swapFirstFixingDate), hedge_times + 1)
    prd = BarrierPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        barrier=torch.tensor(20000),
        obsTL=dTL,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional,
        smooth=torch.tensor(0.05),
        smoothing='sigmoid'
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
            cPrd = BarrierPayerSwaption(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                barrier=prd.barrier,
                obsTL=dTL[int(t0):],
                swapFirstFixingDate=swapFirstFixingDate - t0,
                swapLastFixingDate=swapLastFixingDate - t0,
                notional=notional,
                smooth=prd.smooth,
                smoothing=prd.smoothing
            )
            cRng = RNG(seed=seed, use_av=use_av)
            payoffs = mcSim(cPrd, cMdl, cRng, len(r0_vec))
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res

    """ Delta Hedge Experiment """
    """ Calculate `true` swaption price using Monte Carlo for comparison """
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            swap, swpt = pickle.load(file)
    else:
        r0_test_vec = torch.linspace(r0_min, r0_max, N_test)
        swpt = torch.full_like(r0_test_vec, torch.nan)
        for j in tqdm(range(len(r0_test_vec))):
            tmp_mdl = Vasicek(a, b, sigma, r0_test_vec[j], use_ATS=True, use_euler=False, measure='terminal')
            tmp_rng = RNG(seed=seed, use_av=True)
            swpt[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 500000)))
        swpt = swpt.reshape(-1, 1)

        swap = tmp_mdl.calc_swap(r0_test_vec, t_swap_fixings, delta, strike, notional)

        with open(file_path, 'wb') as file:
            pickle.dump(tuple([swap, swpt]), file, pickle.HIGHEST_PROTOCOL)

    # Initialize experiment
    #swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

    V = swpt.flatten()


    h_a = diff_reg_fit_predict(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                               calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                               diff_reg=diff_reg, use_av=use_av)[1].flatten()
    h_b = V - h_a * swap.flatten()

    # Simulate paths
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

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
            h_a = diff_reg_fit_predict(u_vec=swap, r0_vec=r0_vec, t0=t,
                                       calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                       diff_reg=diff_reg, use_av=use_av)[1].flatten()
            h_b = V - h_a * swap

    swapT = torch.linspace(float(swap.min()), float(swap.max()), 1001)
    payoff_func = max0(swapT) * torch.where(swapT <= prd.barrier, 1.0, 0.0)

    MAE_value = torch.mean(torch.abs(V - max0(swap)))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots()
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
    fig.legend(by_label.values(), by_label.keys(), draggable=True, ncol=2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    #plt.savefig(get_plot_path('vasicek_AAD_DiffReg_delta_hedge_BarrierPayerSwpt.png'), dpi=400)
    plt.show()

