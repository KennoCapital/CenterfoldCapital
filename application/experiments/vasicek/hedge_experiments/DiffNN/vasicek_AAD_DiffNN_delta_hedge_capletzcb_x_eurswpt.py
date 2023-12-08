import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from tqdm import tqdm
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import CapletAsPutOnZCB, EuropeanPayerSwaption
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.path_config import get_plot_path
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import calc_delta_diff_nn

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024 * 4
    N_test = 256
    use_av = True

    hedge_points = 100

    r0_min = -0.02
    r0_max = 0.15

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = 256 * 10
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4
    nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                 'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_size,
                 'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

    # Model specification
    r0 = torch.linspace(r0_min, r0_max, N_test)
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = torch.tensor(0.0871) #mdl.calc_swap_rate(r0.median(), exerciseDate, delta)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    # Simulate paths
    dTL = torch.linspace(0.0, float(exerciseDate), hedge_points + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

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
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='risk_neutral')
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL > exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec), cTL)
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res


    """ Delta Hedge Experiment """

    cpl = mdl.calc_cpl(r0, exerciseDate, delta, strike, notional)[0]

    # Initialize experiment
    B = torch.ones((N_test, ))
    zcb = mdl.calc_zcb(r0, exerciseDate + delta)[0]

    V = cpl
    h_a = calc_delta_diff_nn(u_vec=zcb, t0=0.0, r0_vec=r0_vec,
                               calc_dU_dr=calc_dzcb_dr, calc_dPrd_dr=calc_dcpl_dr,
                               nn_Params=nn_params, use_av=use_av)
    h_b = (V - h_a * zcb) / B


    print("Delta Hedging Caplet")
    for k in tqdm(range(1, len(dTL))):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        # Update market variables
        zcb = mdl.calc_zcb(r[k, :], exerciseDate - t + delta)[0]

        # Update portfolio
        B *= torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        V = h_a * zcb + h_b * B

        if k < len(dTL) - 1:
            h_a = calc_delta_diff_nn(u_vec=zcb, r0_vec=r0_vec, t0=t,
                                       calc_dU_dr=calc_dzcb_dr, calc_dPrd_dr=calc_dcpl_dr,
                                       nn_Params=nn_params, use_av=use_av)
            h_b = (V - h_a * zcb) / B

    zcbT = mdl.calc_zcb(r[-1, ].sort().values, delta)[0]
    df = mdl.calc_zcb(r[-1, ].sort().values, delta)[0]
    K_bar = 1.0 + delta * strike
    payoff_func = notional * K_bar * max0(1.0 / K_bar - zcbT) * df

    V *= mdl.calc_zcb(r[-1], delta)[0]

    MAE_value = torch.mean(torch.abs(V - notional * K_bar * max0(1.0 / K_bar - mdl.calc_zcb(r[-1, :], delta)[0])))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].plot(zcbT, payoff_func, color='black', label='Payoff function')
    ax[0].plot(mdl.calc_zcb(r[-1], delta)[0], V, 'o', color='orange', label='Value of Hedge Portfolio', alpha=0.5)
    ax[0].set_xlabel('P(T, T+delta)')
    ax[0].text(0.8, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax[0].transAxes)

    # Adjust size of plot
    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    ax[0].set_title(f'1Y3M Caplet on Zero Coupon Bond')


    """EUR SWAPTION"""
    r0 = torch.linspace(r0_min, r0_max, N_test)
    mdl.r0 = r0

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

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    # Simulate paths
    dTL = torch.linspace(0.0, float(swapFirstFixingDate), hedge_points + 1)
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
    print("Generating MC prices")
    for n in tqdm(range(N_test)):
        mdl.r0 = r[0, n]
        swpt[n] = torch.mean(mcSim(prd, mdl, rng, 500000))

    # Initialize experiment
    swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

    V = swpt
    h_a = calc_delta_diff_nn(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                             calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                             nn_Params=nn_params, use_av=use_av)
    h_b = V - h_a * swap

    print("Delta Hedging Swaption")
    # Loop over time
    for k in tqdm(range(1, len(dTL))):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        # Update market variables
        swap = mdl.calc_swap(r[k, :], t_swap_fixings - t, delta, strike, notional)

        # Update portfolio
        V = h_a * swap + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        if k < len(dTL) - 1:
            r0_vec = choose_training_grid(r[k, :], N_train)
            h_a = calc_delta_diff_nn(u_vec=swap, r0_vec=r0_vec, t0=t,
                                     calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                     nn_Params=nn_params, use_av=use_av)
            h_b = V - h_a * swap

    swapT = torch.linspace(float(swap.min()), float(swap.max()), 1001)
    payoff_func = max0(swapT)

    MAE_value = torch.mean(torch.abs(V - max0(swap)))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    ax[1].plot(swapT, payoff_func, color='black', label='Payoff function')
    ax[1].plot(swap, V, 'o', color='orange', label='Value of Hedge Portfolio', alpha=0.5)
    ax[1].set_xlabel('Swap(T)')
    ax[1].text(0.05, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax[1].transAxes)

    # Adjust size of plot
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    ax[1].set_title('1Y6Y3M European Payer Swaption')

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.93))

    #plt.savefig(get_plot_path('vasicek_AAD_DiffReg_delta_hedge_capletzcb_x_eurswpt.png'), dpi=400)
    plt.show()
