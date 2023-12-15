import concurrent.futures
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

MAX_PROCESSES = os.cpu_count() - 1
print(f"MAX PROCESSES SETTING={MAX_PROCESSES}")

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
    lam, N_train, batch_ratio, steps, mdl, prd, rng, N_test = args
    r0 = torch.tensor(0.08)  # torch.linspace(r0_min, r0_max, N_test)
    mdl.r0 = r0

    min_batch_size = N_train // batch_ratio
    nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                 'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_size,
                 'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

    # Simulate paths
    # hedge_error = []

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

    # normal hedge error
    # hedge_error.append(torch.std(V - max0(swap)))
    # out[lam, N_train, i, steps] = torch.std(V - max0(swap))
    hedge_error = torch.std(V - max0(swap))
    return lam, N_train, min_batch_size, steps, hedge_error


if __name__ == '__main__':

    #start_time = time.time()

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
    b = torch.tensor(0.09) #r0.median()
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

    strike = torch.tensor(.0871) #mdl.calc_swap_rate(r0.median(), t_swap_fixings, delta)

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
    min_batch_size = 256
    #lam = 1.0
    hidden_units = 20
    hidden_layers = 4

    # varying sets
    lams = [0.0, 1.0]
    training_sets = [1024] #, 1024 * 8]
    batch_ratios = [2, 4] #, 8] #, 8])
    hedge_times = [1, 2, 4, 12] #, 250 // 5]  # , 250//2, 250]

    import itertools
    combinations = list(itertools.product(lams, training_sets, batch_ratios, hedge_times))

    #for index, combo in enumerate(combinations):
    #    lam, N_train, batch_ratio, steps = combo
    #    print(f"Index: {index}, Lambda: {lam}, N_train: {N_train}, Batch ratio: {batch_ratio}, Hedge time: {steps}")

    args = [(lam, N_train, batch_ratio, steps, mdl, prd, rng, N_test)
        for lam, N_train, batch_ratio, steps in combinations]
    with ProcessPoolExecutor(MAX_PROCESSES) as executor:
        results = executor.map(hedge, args) #executor.map(parrallel_tasks, *zip(*combinations))
        for result in results:
            print(result)


    """
    # use parallel computing
    out = torch.zeros((len(lams), len(training_sets), len(batch_ratios), len(hedge_times)))
    def parallel_tasks(lam, N_train):
        min_batch_sizes = N_train // batch_ratios
        for i in range(len(min_batch_sizes)):
            r0 = torch.tensor(0.08)  # torch.linspace(r0_min, r0_max, N_test)
            mdl.r0 = r0

            nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                         'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_sizes[i],
                         'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

            # Simulate paths
            #hedge_error = []

            for steps in range(len(hedge_times)):
                print(f'Frequency={hedge_times[steps]}, N_train={N_train}, lambda={lam} and batch size={min_batch_sizes[i]}')

                dTL = torch.linspace(0.0, float(swapFirstFixingDate), hedge_times[steps] + 1)
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
                        r0_vec = choose_training_grid(r[k, :], N_train)
                        h_a = calc_delta_diff_nn(u_vec=swap, r0_vec=r0_vec, t0=t,
                                                 calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                                 nn_Params=nn_params, use_av=use_av)
                        h_b = V - h_a * swap
                # price adjusted
                # hedge_error.append(torch.std((V - max0(swap))/swpt))

                # normal hedge error
                #hedge_error.append(torch.std(V - max0(swap)))
                #out[lam, N_train, i, steps] = torch.std(V - max0(swap))
                return torch.std(V - max0(swap))


    plt.figure()

    tasks = [(lam, N_train) for lam in range(len(lams)) for N_train in range(len(training_sets))]
    with ProcessPoolExecutor(MAX_PROCESSES) as executor:
        results = executor.map(lambda x: parallel_tasks(*x), tasks)
        for item, result in zip(results):
            print(item, result)




    for lam in lams:
        for N_train in tqdm(training_sets):

            line, = plt.plot(np.log(hedge_times),
                             np.log(hedge_error),
                             'o-',
                             label=f'N={N_train}, bsz={min_batch_sizes[i]}',
                             color=random_color(lam))
            xytext_position = (-50, 0) if lam == 0.0 else (50, 0)
            plt.annotate(f'N={N_train}, bsz={min_batch_sizes[i]}',
                         xy=(np.log(hedge_times[0]), np.log(hedge_error[0])) if lam == 0.0 else (
                             np.log(hedge_times[-1]), np.log(hedge_error[-1])),
                         xytext=xytext_position,
                         textcoords='offset points',
                         color=line.get_color()
                         )

    av_str = 'with AV' if use_av else 'without AV'

    plt.title(prd.name + f'epochs = {epochs} ' + av_str)
    plt.xlabel('Hedge Frequency')
    #plt.ylabel('Std. of Price Adjusted Hedge Error')
    plt.ylabel('Std. of Hedge Error')
    #plt.xticks(ticks=hedge_times, labels=hedge_times)
    plt.xticks(ticks=np.log(hedge_times), labels=hedge_times)
    plt.savefig(get_plot_path('vasicek_AAD_DiffNN_delta_hedge_EuPayerSwpt_convergence_hedgetimes_new.png'), dpi=400)
    plt.show()

    end_time = time.time()
    runtime = (end_time - start_time) / 60
    print(f"Runtime of the script is {runtime} minutes")
    """