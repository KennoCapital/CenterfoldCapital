import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import EuropeanPayerSwaption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import diff_reg_fit_predict, log_plotter

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

    r0_min = -0.02
    r0_max = 0.15

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Model specification
    r0 = torch.tensor(0.08) #torch.linspace(r0_min, r0_max, N_test)
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
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

    strike = torch.tensor(0.0871)

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

    # Setup Differential Regressor, and Scalar
    alpha = 1.0
    deg = 7
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=True)

    degrees = [3, 5, 7, 9]
    degree_colors = {3: "red",
                     5: "orange",
                     7: "darkred",
                     9: "pink"
                     }

    fig, ax = plt.subplots(1, 2)

    for deg in tqdm(degrees, desc="Generating for Hedge Frequency"):
        diff_reg.deg = deg

        # Simulate paths
        hedge_times = [1, 2, 4, 12, 250//5, 250//2, 250, 250*2]
        hedge_error = []

        for steps in hedge_times:

            dTL = torch.linspace(0.0, float(swapFirstFixingDate), steps + 1)
            mcSimPaths(prd, mdl, rng, N_test, dTL)
            r = mdl.x

            # Get price of claim (we use 500k simulations to get an accurate estimate)
            if r0.dim() != 0:
                swpt = torch.empty_like(r[0, :])
                for n in range(N_test):
                    mdl.r0 = r[0, n]
                    swpt[n] = torch.mean(mcSim(prd, mdl, rng, 50000))
            else:
                price = torch.mean(mcSim(prd, mdl, rng, 500000))
                swpt = price * torch.ones(N_test)

            # Initialize experiment
            swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

            V = swpt
            h_a = diff_reg_fit_predict(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                                       calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                       diff_reg=diff_reg, use_av=use_av)[1].flatten()
            h_b = V - h_a * swap

            # Loop over time
            for k in range(1, len(dTL)):
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

            hedge_error.append(torch.std(V - max0(swap)))


        ax[0].plot(np.log(hedge_times), np.log(hedge_error), 'o-', label=f'Deg={deg}', color=degree_colors[deg])
        ax[0].annotate(f'Deg={deg}',
                     xy=(np.log(hedge_times[-1]), np.log(hedge_error[-1])),
                     textcoords='offset points',
                     color=degree_colors[deg],
                     horizontalalignment='center'
                     )

    ax[0].set_title(f'Samples={N_train}')
    ax[0].set_xlabel('Hedge Frequency')
    ax[0].set_ylabel('Std. of Hedge Error')
    ax[0].set_xticks(ticks=np.log(hedge_times), labels=hedge_times)


    """Training Samples"""
    steps = 250
    dTL = torch.linspace(0.0, float(swapFirstFixingDate), steps + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    # Setup Differential Regressor, and Scalar
    for deg in tqdm(degrees, desc="Generating for Varying Training Size"):
        diff_reg.deg = deg

        # Simulate paths
        N_train = [256, 512, 1024, 4096, 8192, 16384]
        hedge_error = []

        for N in N_train:

            r0_vec = torch.linspace(r0_min, r0_max, N)

            # Get price of claim (we use 500k simulations to get an accurate estimate)
            if r0.dim() != 0:
                swpt = torch.empty_like(r[0, :])
                for n in range(N_test):
                    mdl.r0 = r[0, n]
                    swpt[n] = torch.mean(mcSim(prd, mdl, rng, 50000))
            else:
                price = torch.mean(mcSim(prd, mdl, rng, 500000))
                swpt = price * torch.ones(N_test)

            # Initialize experiment
            swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

            V = swpt * torch.ones_like(r[0, :])
            h_a = diff_reg_fit_predict(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                                       calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                       diff_reg=diff_reg, use_av=use_av)[1].flatten()
            h_b = V - h_a * swap

            # Loop over time
            for k in range(1, len(dTL)):
                dt = dTL[k] - dTL[k - 1]
                t = dTL[k]

                # Update market variables
                swap = mdl.calc_swap(r[k, :], t_swap_fixings - t, delta, strike, notional)

                # Update portfolio
                V = h_a * swap + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
                if k < len(dTL) - 1:
                    r0_vec = choose_training_grid(r[k, :], N)
                    h_a = diff_reg_fit_predict(u_vec=swap, r0_vec=r0_vec, t0=t,
                                               calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                               diff_reg=diff_reg, use_av=use_av)[1].flatten()
                    h_b = V - h_a * swap

            hedge_error.append(torch.std(V - max0(swap)))
        ax[1].plot(np.log(N_train), np.log(hedge_error), 'o-', label=f'Deg={deg}', color=degree_colors[deg])
        ax[1].annotate(f'Deg={deg}',
                     xy=(np.log(N_train[-1]), np.log(hedge_error[-1])),
                     textcoords='offset points',
                     color=degree_colors[deg],
                     horizontalalignment='center'
                     )

    ax[1].set_title(f'Hedge Times={steps}')
    ax[1].set_xlabel('Training Sample')
    ax[1].set_xticks(ticks=np.log(N_train), labels=N_train)

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.suptitle(prd.name)

    plt.show()