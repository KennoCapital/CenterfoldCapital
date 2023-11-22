import torch
import matplotlib.pyplot as plt
import numpy as np
from application.engine.differential_NN import Neural_Approximator
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import EuropeanPayerSwaption
from application.engine.mcBase import mcSim, RNG

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


if __name__ == '__main__':

    seed = 1234
    N_train = 256

    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 50
    batches_per_epoch = 16
    min_batch_size = 128 #256 * 4
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    use_av = True
    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(.25)
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

    """ Estimate Delta using Differential Regression """

    # Make helper functions
    def calc_dswap_dr(x, s):
        """
        :param  x:    Short rate r0
        :param  s:    Current time

        returns:
            tuple with: (Swap Prices, Swap Prices differentiated wrt. r0 evaluated at x)
        """
        def _swap_price(x):
            tau = t_swap_fixings - s
            S = mdl.calc_swap(x, tau, delta, strike, notional)
            return S
        ones = torch.ones_like(x)
        res = jvp(_swap_price, x, ones, create_graph=False)
        return res

    def calc_dswpt_dr(x, s):
        """
        :param  x:    Short rate r0
        :param  s:    Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        def _payoffs(x):
            cMdl = Vasicek(a, b, sigma, x, use_ATS=True, use_euler=False, measure=measure)
            cPrd = EuropeanPayerSwaption(
                    strike=strike,
                    exerciseDate=exerciseDate - s,
                    delta=delta,
                    swapFirstFixingDate=swapFirstFixingDate - s,
                    swapLastFixingDate=swapLastFixingDate - s,
                    notional=notional
            )
            payoffs = mcSim(cPrd, cMdl, rng, N_train)
            return payoffs

        ones = torch.ones_like(x)
        res = jvp(_payoffs, x, ones, create_graph=False)
        return res

    """ Calculate MC swaption price against r0 and swap(0) """
    r_grid = torch.linspace(0.00, 0.15, 101)
    swpt_grid = torch.full_like(r_grid, torch.nan)
    for j in range(len(r_grid)):
        tmp_mdl = Vasicek(a, b, sigma, r_grid[j], use_ATS=True, use_euler=False, measure='terminal')
        tmp_rng = RNG(seed=seed, use_av=True)
        swpt_grid[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 50000)))

    swap_grid = tmp_mdl.calc_swap(r_grid, t_swap_fixings, delta, strike, notional)
    dswpt_dswap = swpt_grid.diff() / swap_grid.diff()

    """ Plot Differential Neutral Network (in sample) """

    r0_grid = torch.linspace(0.00, 0.15, N_train // 2)
    r0_grid = torch.concat([r0_grid, r0_grid])

    swap, dSdr = calc_dswap_dr(r0_grid, 0.0)
    y, dydr = calc_dswpt_dr(r0_grid, 0.0)

    X_train = swap.reshape(-1, 1)
    y_train = y.reshape(-1, 1)
    z_train = (dydr / dSdr).reshape(-1, 1)

    # AV-reduction the right way
    idx_half = N_train // 2
    X_train = X_train[:idx_half]
    y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
    z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])

    X_test = X_train  # Here X_train is what makes the plot 'in-sample'

    # Setup Differential Neutral Network
    diff_nn = Neural_Approximator(X_train, y_train, z_train)
    diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units, hidden_layers=hidden_layers)
    diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
    y_pred, z_pred = diff_nn.predict_values_and_derivs(X_test)


    """ Plot results """
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    # Plot price function
    ax[0, 1].plot(X_train.flatten(), y_train.flatten(), 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[0, 1].plot(X_test.flatten(), y_pred.flatten(), label='Predictions', color='orange')
    ax[0, 1].plot(swap_grid, swpt_grid, color='black', label='Analytical (Bump and reval)')
    ax[0, 1].set_ylabel('Price')

    # Plot delta function
    ax[1, 1].plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[1, 1].plot(X_test, z_pred, label='Predictions', color='orange')
    ax[1, 1].plot(swap_grid[1:], dswpt_dswap, color='black', label='Analytical (Bump and reval)')
    ax[1, 1].set_xlabel('P(0, T + delta)')
    ax[1, 1].set_ylabel('Delta')

    # Adjust size of subplots
    box0 = ax[0, 1].get_position()
    ax[0, 1].set_position([box0.x0, box0.y0 - box0.height * 0.1, box0.width, box0.height * 0.9])

    box1 = ax[1, 1].get_position()
    ax[1, 1].set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])

    # Title
    av_str = 'with AV' if use_av else 'without AV'
    ax[0, 1].set_title(f'Differential Neural Network')



    """STANDARD NEURAL NETWORK"""

    # Setup Standard Neutral Network for comparison
    lam = 0.0

    diff_nn = Neural_Approximator(X_train, y_train, z_train)
    diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                    hidden_layers=hidden_layers)
    diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
    y_pred, z_pred = diff_nn.predict_values_and_derivs(X_test)

    """ Plot results """
    # Plot price function
    ax[0, 0].plot(X_train.flatten(), y_train.flatten(), 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[0, 0].plot(X_test.flatten(), y_pred.flatten(), label='Predictions', color='orange')
    ax[0, 0].plot(swap_grid, swpt_grid, color='black', label='Analytical (Bump and reval)')
    ax[0, 0].set_ylabel('Price')

    # Plot delta function
    ax[1, 0].plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[1, 0].plot(X_test, z_pred, label='Predictions', color='orange')
    ax[1, 0].plot(swap_grid[1:], dswpt_dswap, color='black', label='Analytical (Bump and reval)')
    ax[1, 0].set_xlabel('P(0, T + delta)')
    ax[1, 0].set_ylabel('Delta')

    # Adjust size of subplots
    box0 = ax[0, 0].get_position()
    ax[0, 0].set_position([box0.x0, box0.y0 - box0.height * 0.1, box0.width, box0.height * 0.9])

    box1 = ax[1, 0].get_position()
    ax[1, 0].set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])


    # Title
    av_str = 'with AV' if use_av else 'without AV'
    ax[0, 0].set_title(f'Standard Neural Network')

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), draggable = True, ncol=3, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    fig.suptitle(prd.name + f'\n {N_train} training samples ' + av_str)

    plt.show()








