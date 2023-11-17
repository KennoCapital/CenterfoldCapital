import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import CapletAsPutOnZCB
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSim, RNG
from application.utils.path_config import get_plot_path

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

    r0_min = 0.02
    r0_max = 0.13

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
    notional = torch.tensor(1e6)

    strike = mdl.calc_swap_rate(r0, exerciseDate, delta)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

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
            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec))
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res


    def training_data(r0_vec: torch.Tensor, t0: float = 0.0, use_av: bool = True):
        if use_av:
            # X_train[i] = X_train[i + N_train],  for all i, when using AV
            r0_vec = torch.concat([r0_vec, r0_vec])

        fwd, dSdr = calc_dzcb_dr(r0_vec, t0)
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


    """ Calculate Analytical Caplet price """
    r0_test_vec = torch.linspace(r0_min, r0_max, N_test)
    X_test = mdl.calc_zcb(r0_test_vec, exerciseDate + delta)[0].reshape(-1, 1)
    y_mdl = mdl.calc_cpl(r0_test_vec, exerciseDate, delta, strike, notional)[0].reshape(-1, 1)
    z_mdl = y_mdl.diff(dim=0) / X_test.diff(dim=0)

    plt.figure()
    plt.plot(X_test, z_mdl, 'o')
    plt.show()

    """ Estimate Price and Delta using Differential Regression """

    X_train, y_train, z_train = training_data(r0_vec=r0_vec, t0=0.0, use_av=use_av)

    X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

    diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

    X_test_scaled, _, _ = scalar.transform(X_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    RMSE_price = torch.sqrt(torch.mean((y_pred - y_mdl) ** 2))
    MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_mdl))

    """ Plot results """


    fig, ax = plt.subplots(2, sharex='col')
    # Plot price function
    ax[0].plot(X_train.flatten(), y_train.flatten(), 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[0].plot(X_test.flatten(), y_pred, label='DiffReg', color='orange')
    ax[0].plot(X_test, y_mdl, color='black', label='Analytical (Bump and reval)')
    ax[0].set_ylabel('Price')
    ax[0].text(0.05, 0.8, f'RMSE = {RMSE_price:.2f}', fontsize=8, transform=ax[0].transAxes)

    # Plot delta function
    ax[1].plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[1].plot(X_test, z_pred, label='DiffReg', color='orange')
    ax[1].plot(X_test[1:], z_mdl, color='black', label='Analytical (Bump and reval)')
    ax[1].set_xlabel('P(0, T + delta)')
    ax[1].set_ylabel('Delta')
    ax[1].text(0.05, 0.8, f'MAE = {MAE_delta:.4f}', fontsize=8, transform=ax[1].transAxes)

    # Adjust size of subplots
    box0 = ax[0].get_position()
    ax[0].set_position([box0.x0, box0.y0 - box0.height * 0.1, box0.width, box0.height * 0.9])

    box1 = ax[1].get_position()
    ax[1].set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    # Title
    av_str = 'with AV' if use_av else 'without AV'
    fig.suptitle(prd.name + f'\nalpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str)

    plt.savefig(get_plot_path('vasicek_AAD_DiffReg_plot_delta_caplet_as_put_on_zcb.png'), dpi=400)
    plt.show()
