import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import Caplet
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
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

    hedge_points = 25

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
    exerciseDate = torch.tensor(0.25)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = mdl.calc_swap_rate(r0, exerciseDate, delta)

    prd = Caplet(
        strike=strike,
        start=exerciseDate,
        delta=delta
    )

    # Simulate paths
    dTL = torch.linspace(0.0, float(exerciseDate), hedge_points + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    # Find index of the exercise date
    last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])

    """ Helper functions for calculating pathwise payoffs and deltas, and generating training data """
    def calc_dfwd_dr(r0_vec, t0):
        """
        :param  r0_vec:    Current Short rate r0
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """

        def _cpl(r0_vec):
            fwd = mdl.calc_fwd(r0_vec, exerciseDate - t0, delta)[0]
            return fwd

        ones = torch.ones_like(r0_vec)
        res = jvp(_cpl, r0_vec, ones, create_graph=False)
        return res


    def calc_dcpl_dr(r0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """

        def _payoffs(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = Caplet(
                strike=strike,
                start=exerciseDate - t0,
                delta=delta
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

        fwd, dSdr = calc_dfwd_dr(r0_vec, t0)
        y, dydr = calc_dcpl_dr(r0_vec, t0)

        X_train = fwd.reshape(-1, 1)
        y_train = y.reshape(-1, 1)
        z_train = (dydr / dSdr).reshape(-1, 1)

        if use_av:
            idx_half = N_train
            X_train = X_train[:idx_half]
            y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
            z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])

        return X_train, y_train, z_train

    def calc_delta(fwd_vec: torch.Tensor, r0_vec: torch.Tensor, t0: float, use_av: bool) -> torch.Tensor:
        """
        param fwd_vec:      1D vector of market variables (Fwd prices)
        param r0_vec:       1D vector of short rates to generate training data from (swap prices)
        param t0:           Current market time, effect time to expiry and fixings in the training
        param use_av:       Use antithetic variates to reduce variance of both y- and z-labels

        returns:            1D vector of predicted deltas for `swap_vec`
        """
        X_test = fwd_vec.reshape(-1, 1)

        X_train, y_train, z_train = training_data(r0_vec=r0_vec, t0=t0, use_av=use_av)

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

        X_test_scaled, _, _ = scalar.transform(X_test, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        return z_pred.flatten()

    def calc_delta_bump_and_reval(r0_vec: torch.Tensor, t0: float, delta: torch.tensor, strike: torch.tensor, bump: float = 0.0001) -> torch.Tensor:
        """

        returns:            1D vector of predicted deltas
        """

        r_bump = r0_vec + bump
        fwd = mdl.calc_fwd(r0_vec, t0, delta)[0]
        fwd_bump = mdl.calc_fwd(r_bump, t0, delta)[0]
        cpl = mdl.calc_cpl(r0_vec, t0, delta, strike)[0]
        cpl_bump = mdl.calc_cpl(r_bump, t0, delta, strike)[0]

        z = (cpl_bump - cpl) / (fwd_bump - fwd)

        return z

    """ Delta Hedge Experiment """

    # Get price of claim (no need to simulate as we have an analytical expression)
    # cpl = mdl.calc_cpl(r0, exerciseDate, delta, strike)[0]
    cpl = torch.mean(mcSim(prd, mdl, rng, 500000))

    # Initialize experiment
    fwd = mdl.calc_fwd(r[0, :], exerciseDate, delta)[0]

    V = cpl * torch.ones_like(r[0, :])
    #h_a = calc_delta_bump_and_reval(r[0], exerciseDate - 0.0, delta, strike)
    h_a = calc_delta(fwd_vec=fwd, r0_vec=r0_vec, t0=0.0, use_av=use_av)
    h_b = V - h_a * fwd

    # Loop over time
    for k in range(1, last_idx + 1):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        # Update market variables
        fwd = mdl.calc_fwd(r[k, :], exerciseDate - t, delta)[0]

        # Update portfolio
        V = h_a * fwd + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        if k < last_idx:
            #h_a = calc_delta_bump_and_reval(r[k, :], exerciseDate - t, delta, strike)
            h_a = calc_delta(fwd_vec=fwd, r0_vec=r0_vec, t0=t, use_av=use_av)
            h_b = V - h_a * fwd

    # fwdT = torch.linspace(float(fwd.min()), float(fwd.max()), 1001)
    fwdT = mdl.calc_fwd(r[last_idx, ].sort().values, torch.tensor(0.0), delta)[0]
    df = mdl.calc_zcb(r[last_idx, ].sort().values, delta)[0]
    payoff_func = delta * max0(fwdT - strike) * df

    MAE_value = torch.mean(torch.abs(V - max0(fwd)))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots(1)
    ax.plot(fwdT, payoff_func, color='black', label='Payoff function')
    ax.plot(fwd, V, 'o', color='orange', label='Value of Hedge Portfolio', alpha=0.5)
    ax.set_xlabel('Fwd(T)')
    ax.text(0.05, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax.transAxes)

    # Adjust size of plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    fig.suptitle(
        prd.name + f'\nHedgeFreq={dTL[1]:.4g}, alpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str)

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    plt.savefig(get_plot_path('vasicek_AAD_DiffReg_Caplet_delta_hedge.png'), dpi=400)
    plt.show()
