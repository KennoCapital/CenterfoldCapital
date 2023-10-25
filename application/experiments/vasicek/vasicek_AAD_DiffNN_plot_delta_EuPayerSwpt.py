import torch
import matplotlib.pyplot as plt

from application.engine.differential_NN import Neural_Approximator
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import EuropeanPayerSwaption
from application.engine.mcBase import mcSim, RNG

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


if __name__ == '__main__':

    seed = 1234
    N_train = 4096 * 2

    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 100
    batches_per_epoch = 16
    min_batch_size = 256 * 4
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

    rng = RNG(seed=seed, use_av=True)

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
    r_grid = torch.linspace(0.03, 0.15, 101)
    swpt_grid = torch.full_like(r_grid, torch.nan)
    for j in range(len(r_grid)):
        tmp_mdl = Vasicek(a, b, sigma, r_grid[j], use_ATS=True, use_euler=False, measure='terminal')
        tmp_rng = RNG(seed=seed, use_av=True)
        swpt_grid[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 50000)))

    swap_grid = tmp_mdl.calc_swap(r_grid, t_swap_fixings, delta, strike, notional)
    dswpt_dswap = swpt_grid.diff() / swap_grid.diff()

    """ Plot Differential Neutral Network (in sample) """

    r0_grid = torch.linspace(0.03, 0.15, N_train // 2)
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

    plt.figure()
    plt.plot(X_train, y_train, 'o', color='gray', alpha=0.25, label='Sample Payoffs')
    plt.plot(X_test, y_pred, label='DiffNN', color='orange')
    plt.plot(swap_grid, swpt_grid, color='black', label='bump and reval')
    plt.title('Learning Payoffs')
    plt.xlabel('Swap(0)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Sample Differentials')
    plt.plot(X_test, z_pred, label='DiffNN', color='orange')
    plt.plot(swap_grid[1:], dswpt_dswap, color='black', label='bump and reval')
    plt.xlabel('Swap(0)')
    plt.title('Learning Sensitivities')
    plt.legend()
    plt.show()
