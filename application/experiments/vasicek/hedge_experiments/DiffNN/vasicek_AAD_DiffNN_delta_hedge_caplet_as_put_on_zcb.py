import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from tqdm import tqdm
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import CapletAsPutOnZCB
from application.experiments.vasicek.vasicek_hedge_tools import calc_delta_diff_nn
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.path_config import get_plot_path
from application.utils.torch_utils import max0

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True
    save_plot = False

    # Hedge experiment settings
    hedge_points = 250
    r0_min = -0.02
    r0_max = 0.15
    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = int(N_train * 5 / 8) #256 * 10
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4
    nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                 'batches_per_epoch': batches_per_epoch,'min_batch_size': min_batch_size,
                 'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.linspace(r0_min, r0_max, N_test) #torch.tensor(0.08)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(.25)
    notional = torch.tensor(1e6)

    strike = torch.tensor(0.0871) #mdl.calc_swap_rate(r0, exerciseDate, delta)

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

    # Find index of the exercise date
    last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])

    """ Helper functions (AAD friendly) for calculating pathwise payoffs and deltas, and generating training data """
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

    # Get price of claim (no need to simulate as we have an analytical expression)
    cpl = mdl.calc_cpl(r0, exerciseDate, delta, strike, notional)[0]

    # Initialize experiment
    B = torch.ones((N_test, ))
    zcb = mdl.calc_zcb(r0, exerciseDate + delta)[0]

    V = cpl
    h_a = calc_delta_diff_nn(u_vec=zcb, r0_vec=r0_vec, t0=0.0,
                             calc_dPrd_dr=calc_dcpl_dr, calc_dU_dr=calc_dzcb_dr,
                             nn_Params=nn_params, use_av=use_av)
    h_b = (V - h_a * zcb) / B

    cpl_prices = [V]
    V_values = [V]

    # Loop over time
    for k in tqdm(range(1, last_idx + 1)):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        # Update market variables
        zcb = mdl.calc_zcb(r[k, :], exerciseDate - t + delta)[0]

        # Update portfolio
        B *= torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        V = h_a * zcb + h_b * B

        V_values.append(V)
        cpl_prices.append(mdl.calc_cpl(r[k, :], exerciseDate-t, delta, strike, notional))

        if k < last_idx:
            r0_vec = choose_training_grid(r[k, :], N_train)
            h_a = calc_delta_diff_nn(u_vec=zcb, r0_vec=r0_vec, t0=t,
                                     calc_dPrd_dr=calc_dcpl_dr, calc_dU_dr=calc_dzcb_dr,
                                    nn_Params=nn_params, use_av=use_av)
            h_b = (V - h_a * zcb) / B

    V_values = torch.vstack(V_values)
    cpl_prices = torch.vstack(cpl_prices)

    RMSE = torch.sqrt(torch.mean((V_values - cpl_prices) ** 2, dim=1))
    plt.figure()
    plt.plot(dTL, RMSE)
    plt.xlabel('t')
    plt.ylabel('RMSE')
    plt.show()

    zcbT = mdl.calc_zcb(r[last_idx, ].sort().values, delta)[0]
    df = mdl.calc_zcb(r[last_idx, ].sort().values, delta)[0]
    K_bar = 1.0 + delta * strike
    payoff_func = notional * K_bar * max0(1.0 / K_bar - zcbT) * df

    V *= mdl.calc_zcb(r[last_idx], delta)[0]

    MAE_value = torch.mean(torch.abs(V - notional * K_bar * max0(1.0 / K_bar - mdl.calc_zcb(r[last_idx, :], delta)[0])))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots(1)
    ax.plot(zcbT, payoff_func, color='black', label='Payoff function')
    ax.plot(mdl.calc_zcb(r[last_idx], delta)[0], V, 'o', color='orange', label='Value of Hedge Portfolio', alpha=0.5)
    ax.set_xlabel('ZCB(T)')
    ax.text(0.05, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax.transAxes)

    # Adjust size of plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    fig.suptitle(
        prd.name + f'\nHedgeFreq={dTL[1]:.4g}, epochs = {epochs}, nw={hidden_layers}x{hidden_units}, {N_train} training samples ' + av_str)

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))
    if save_plot:
        plt.savefig(get_plot_path('vasicek_AAD_DiffNN_delta_hedge_caplet.png'), dpi=400)
    plt.show()
