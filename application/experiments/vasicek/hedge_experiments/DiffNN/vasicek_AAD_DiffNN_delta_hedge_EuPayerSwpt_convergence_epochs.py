import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import EuropeanPayerSwaption
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.path_config import get_plot_path
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import calc_delta_diff_nn

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    start_time = time.time()

    N_train = 1024
    seed = 1234
    N_test = 256
    use_av = True

    r0_min = 0.02
    r0_max = 0.12
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

    strike = torch.tensor(0.0871) #mdl.calc_swap_rate(r0.median(), t_swap_fixings, delta)

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
    steps = 250 // 2
    dTL = torch.linspace(0.0, float(swapFirstFixingDate), steps + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    # Setup Differential Regressor, and Scalar
    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = [10, 25, 50, 100, 250, 500]
    batches_per_epoch = 16
    hidden_units = 20
    hidden_layers = 4
    lam = 1.0
    min_batch_size = int(N_train * 5/8)

    hedge_error = []

    for epoch in tqdm(epochs):
        nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epoch,
                     'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_size,
                     'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

        # Simulate paths
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
        swap = mdl.calc_swap(mdl.r0, t_swap_fixings, delta, strike, notional)

        V = swpt

        h_a = calc_delta_diff_nn(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                             calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                             nn_Params=nn_params, use_av=use_av)
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
                #r0_vec = choose_training_grid(r[k, :], N)
                h_a = calc_delta_diff_nn(u_vec=swap, r0_vec=r0_vec, t0=t,
                                     calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                     nn_Params=nn_params, use_av=use_av)
                h_b = V - h_a * swap

        hedge_error.append(torch.std(V - max0(swap)))

    plt.figure()
    plt.plot(np.log(epochs), np.log(hedge_error), 'o-', color='orange')
    plt.title(prd.name + f', Hedge Times={steps} , Notional = {int(notional)}')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Std. of Hedge Error')
    plt.xticks(ticks=np.log(epochs), labels=epochs)
    plt.show()
