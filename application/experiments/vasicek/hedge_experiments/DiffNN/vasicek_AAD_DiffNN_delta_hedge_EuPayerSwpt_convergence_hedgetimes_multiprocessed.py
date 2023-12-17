import concurrent.futures

import pandas
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import os
import itertools
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

MAX_PROCESSES = os.cpu_count() - 1


def random_color(lam : float):
    """Generate random colors within specific ranges based on the value of lam."""
    if lam == 1.0:
        # Generate orange-like colors
        return "#{:02x}{:02x}{:02x}".format(random.randint(200, 255), random.randint(100, 160), random.randint(0, 50))
    elif lam == 0.0:
        # Generate darker colors
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    else:
        # Default color if lam is neither 1.0 nor 0.0
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def parrallel_tasks(lam, N_train, batch_ratio, steps):
    return lam, N_train, batch_ratio, steps

def hedge(args):
    lam, N_train, batch_ratio, steps = args

    seed = 1234
    N_test = 256
    use_av = True

    r0_min = -0.02
    r0_max = 0.15
    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Model specification
    r0 = torch.tensor(0.08)
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)
    rng.gen = torch.Generator().manual_seed(seed)

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

    strike = torch.tensor(.0871)

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
    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 50
    batches_per_epoch = 16
    hidden_units = 20
    hidden_layers = 4
    min_batch_size = N_train // batch_ratio

    nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                 'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_size,
                 'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

    # simulate state variable
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
    h_a = calc_delta_diff_nn(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                             calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                             nn_Params=nn_params, use_av=use_av)
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
            #r0_vec = choose_training_grid(r[k, :], N_train)
            h_a = calc_delta_diff_nn(u_vec=swap, r0_vec=r0_vec, t0=t,
                                     calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                     nn_Params=nn_params, use_av=use_av)
            h_b = V - h_a * swap
    # price adjusted
    # hedge_error.append(torch.std((V - max0(swap))/swpt))

    hedge_error = torch.std(V - max0(swap))
    return lam, N_train, min_batch_size, steps, hedge_error


if __name__ == '__main__':

    start_time = time.time()

    # varying sets
    lams = [0.0, 1.0]
    training_sets = [1024, 1024 * 8, 1024 * 16]
    batch_ratios = [1, 2, 4, 8]
    hedge_times = [1, 2, 4, 12, 250 // 5, 250//2] #, 250] #TODO: Do it with a year and perhaps two years

    combinations = list(itertools.product(lams, training_sets, batch_ratios, hedge_times))

    args = [(lam, N_train, batch_ratio, steps) for lam, N_train, batch_ratio, steps in combinations]
    out = []
    with ProcessPoolExecutor(MAX_PROCESSES) as executor:
        results = executor.map(hedge, args) #executor.map(parrallel_tasks, *zip(*combinations))
        for result in tqdm(results):
            out.append(result)

    import pandas as pd
    df = pandas.DataFrame(out, columns=['LAM', 'N_TRAIN', 'MIN_BATCH_SIZE', 'STEPS', 'HEDGE_ERROR'])
    plt.figure()
    for lam in df['LAM'].unique():
        for n in df['N_TRAIN'].unique():
            for b in df['MIN_BATCH_SIZE'].unique():
                df_tmp = df[(df['LAM'] == lam) & (df['N_TRAIN'] == n) & (df['MIN_BATCH_SIZE'] == b)]
                if not df_tmp.empty:
                    line, = plt.plot(np.log(hedge_times), np.log(df_tmp['HEDGE_ERROR'].values),
                             'o-', label=f'N={n}, bsz={b}', color='black' if lam==0 else 'orange')
                    # set labels correctly
                    if n==1024:
                        y_offset = 15
                    elif n==1024*8:
                        y_offset = 0
                    elif n==16384:
                        y_offset = -15
                    else:
                        raise ValueError("Chose wrong training size!")
                    plt.annotate(f'N={n}, bsz={b}',
                                 xy=(np.log(df_tmp['STEPS'].values[3]), np.log(df_tmp['HEDGE_ERROR'].values[0])) if lam == 0.0
                                    else
                                    (np.log(df_tmp['STEPS'].values[-1]), np.log(df_tmp['HEDGE_ERROR'].values[-1])),
                                 xytext=(0, y_offset),
                                 textcoords='offset points',
                                 color='black' if lam==0 else 'orange',
                                 horizontalalignment='center'
                                 )
    plt.xticks(ticks=np.log(hedge_times), labels=hedge_times)
    plt.xlabel('Hedge Frequency')
    plt.ylabel('Std. of Hedge Error')
    plt.show()

    end_time = time.time()
    runtime = (end_time - start_time) / 60
    print(f"Runtime of the script is {runtime} minutes")
