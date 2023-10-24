import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import EuropeanPayerSwaption
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.torch_utils import max0
from application.utils.path_config import get_plot_path

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

    steps_per_fixing = 10

    r0_min = -0.02
    r0_max = 0.15

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Setup Differential Regressor, and Scalar
    deg = 15
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
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

    strike = mdl.calc_swap_rate(r0, t_swap_fixings, delta)

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    # Simulate paths
    dTL = torch.linspace(0.0, float(swapLastFixingDate), steps_per_fixing * int(swapLastFixingDate / delta) + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    # Find index of the exercise date
    last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])

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

    def training_data(r0_vec: torch.Tensor, t0: float = 0.0, use_av: bool = True):
        if use_av:
            # X_train[i] = X_train[i + N_train],  for all i, when using AV
            r0_vec = torch.concat([r0_vec, r0_vec])

        swap, dSdr = calc_dswap_dr(r0_vec, t0)
        y, dydr = calc_dswpt_dr(r0_vec, t0)

        X_train = swap.reshape(-1, 1)
        y_train = y.reshape(-1, 1)
        z_train = (dydr / dSdr).reshape(-1, 1)

        if use_av:
            idx_half = N_train
            X_train = X_train[:idx_half]
            y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
            z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])

        return X_train, y_train, z_train

    def calc_delta(swap_vec: torch.Tensor, r0_vec: torch.Tensor, t0: float, use_av: bool) -> torch.Tensor:
        """
        param swap_vec:     1D vector of market variables (swap prices)
        param r0_vec:       1D vector of short rates to generate training data from (swap prices)
        param t0:           Current market time, effect time to expiry and fixings in the training
        param use_av:       Use antithetic variates to reduce variance of both y- and z-labels

        returns:            1D vector of predicted deltas for `swap_vec`
        """
        X_test = swap_vec.reshape(-1, 1)

        X_train, y_train, z_train = training_data(r0_vec=r0_vec, t0=t0, use_av=use_av)

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

        X_test_scaled, _, _ = scalar.transform(X_test, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        return z_pred.flatten()

    """ Delta Hedge Experiment """

    # Get price of claim (we use 500k simulations to get an accurate estimate)
    swpt = torch.mean(mcSim(prd, mdl, rng, 500000))

    # Initialize experiment
    swap = mdl.calc_swap(r[0, :], t_swap_fixings, delta, strike, notional)

    V = swpt * torch.ones_like(r[0, :])
    h_a = calc_delta(swap_vec=swap, r0_vec=r0_vec, t0=0.0, use_av=use_av)
    h_b = V - h_a * swap

'''
    u_swap = torch.full((last_idx + 1, N_test), torch.nan)
    u_fwd = torch.full((last_idx + 1, N_test), torch.nan)
    u_s = torch.full((last_idx + 1, N_test), torch.nan)
    u_swap[0] = swap
    u_fwd[0] = mdl.calc_fwd(r0, exerciseDate + 10.0, delta)[0]
    u_s[0] = mdl.calc_swap_rate(r0, t_swap_fixings, delta)
'''
    # Loop over time
    for k in range(1, last_idx + 1):
        dt = dTL[k] - dTL[k-1]
        t = dTL[k]

        # Update market variables
        swap = mdl.calc_swap(r[k, :], t_swap_fixings - t, delta, strike, notional)

'''
        u_swap[k] = swap
        u_fwd[k] = mdl.calc_fwd(r[k, :], exerciseDate + 10.0 - t, delta)[0]
        u_s[k] = mdl.calc_swap_rate(r[k, :], t_swap_fixings - t, delta)
'''
        # Update portfolio
        V = h_a * swap + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        if k < last_idx:
            h_a = calc_delta(swap_vec=swap, r0_vec=r0_vec, t0=t, use_av=use_av)
            h_b = V - h_a * swap
'''
    p = 1
    fig, ax = plt.subplots(2, sharex='all')
    ax[0].plot(dTL[:last_idx + 1], r[:last_idx + 1, p], label='r', color='black')
    ax[0].plot(dTL[:last_idx + 1], u_fwd[:, p], label='fwd', color='red')
    ax[0].plot(dTL[:last_idx + 1], u_s[:, p], label='swap_rate', color='blue')
    ax[1].plot(dTL[:last_idx + 1], u_swap[:, p], label='swap', color='blue')
    ax[1].set_xlabel('Time')
    fig.legend()
    plt.show()
'''
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

    plt.savefig(get_plot_path('vasicek_AAD_DiffReg_EuPaySwpt_delta_hedge.png'), dpi=400)
    plt.show()

