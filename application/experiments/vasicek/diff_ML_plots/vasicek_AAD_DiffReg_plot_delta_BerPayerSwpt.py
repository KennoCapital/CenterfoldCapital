import torch
import pickle
import os
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import BermudanPayerSwaption
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import RNG, LSMC, lsmcDefaultSim
from application.engine.regressor import PolynomialRegressor
from application.utils.path_config import get_plot_path, get_data_path
from tqdm import tqdm
from application.experiments.vasicek.vasicek_hedge_tools import training_data


torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    # Set this to `None` if existing data should not be imported
    test_set_filename = 'vasicek_BerPayerSwpt_test_set'
    filename_path = get_data_path(test_set_filename + '.pkl')

    seed = 1234
    N_train = 16384
    use_av = True

    test_bump_size_bp = torch.tensor(1.0)

    r0_min = torch.tensor(-0.02)
    r0_max = torch.tensor(0.15)

    N_test = torch.round((r0_max - r0_min) * test_bump_size_bp * 10000)

    r0_vec = torch.linspace(float(r0_min), float(r0_max), N_train)

    # Setup Differential Regressor, and Scalar
    deg_lsmc = 15
    deg = 15
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'terminal'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDates = torch.tensor([1.0, 2.0, 5.0])
    delta = torch.tensor(0.25)
    swapFirstFixingDate = torch.tensor(5.0)
    swapLastFixingDate = torch.tensor(10.0)
    notional = torch.tensor(1e6)

    t_swap_fixings = torch.linspace(
        float(swapFirstFixingDate),
        float(swapLastFixingDate),
        int((swapLastFixingDate - swapFirstFixingDate) / delta + 1)
    )

    strike = mdl.calc_swap_rate(r0, t_swap_fixings, delta)

    prd = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    poly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=True)
    lsmc = LSMC(reg=poly_reg)

    """ Helper functions for generating training data of pathwise payoffs and deltas """

    def calc_dswap_dr(r0_vec, t0):
        def _swap_price(r0_vec):
            tau = t_swap_fixings - t0
            S = mdl.calc_swap(r0_vec, tau, delta, strike, notional)
            return S
        ones = torch.ones_like(r0_vec)
        res = jvp(_swap_price, r0_vec, ones, create_graph=False)
        return res

    def calc_dswpt_dr(r0_vec, t0):
        def _swpt(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = BermudanPayerSwaption(
                    strike=strike,
                    exerciseDates=exerciseDates - t0,
                    delta=delta,
                    swapFirstFixingDate=swapFirstFixingDate - t0,
                    swapLastFixingDate=swapLastFixingDate - t0,
                    notional=notional
                )
            cRng = RNG(seed=seed, use_av=use_av)
            cPoly_reg = PolynomialRegressor(deg=deg_lsmc)
            cLsmc = LSMC(reg=cPoly_reg)
            payoff = lsmcDefaultSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec), n=len(r0_vec), lsmc=cLsmc)
            return torch.sum(payoff, dim=0)
        ones = torch.ones_like(r0_vec)
        J = jvp(func=_swpt, inputs=r0_vec, v=ones, create_graph=False)
        return J


    """ Calculate `true` swaption price using Monte Carlo for comparison """
    if test_set_filename is not None and os.path.isfile(filename_path):
        with open(filename_path, "rb") as input_file:
            X_test, y_test, z_test = pickle.load(input_file)
    else:
        r0_test_vec = torch.linspace(float(r0_min), float(r0_max), int(N_test) + 1)
        X_test = mdl.calc_swap(r0_test_vec, t_swap_fixings, delta, strike, notional).reshape(-1, 1)
        y_test = torch.full_like(r0_test_vec, torch.nan)
        for j in tqdm(range(len(r0_test_vec)), desc='Calculating pricing of BerPayerSwpt using MC'):
            tmp_mdl = Vasicek(a, b, sigma, r0_test_vec[j], use_ATS=True, use_euler=False, measure='terminal')
            tmp_rng = RNG(seed=seed, use_av=True)
            payoff = lsmcDefaultSim(prd=prd, mdl=tmp_mdl, rng=tmp_rng, N=500000, n=25000, lsmc=lsmc, reg=poly_reg)
            y_test[j] = torch.mean(torch.sum(payoff, dim=0))
        y_test = y_test.reshape(-1, 1)
        z_test = y_test.diff(dim=0) / X_test.diff(dim=0)

        # Export results
        if test_set_filename is not None:
            with open(filename_path, 'wb') as output_file:
                pickle.dump((X_test, y_test, z_test), output_file, protocol=pickle.HIGHEST_PROTOCOL)

    """ Estimate Price and Delta using Differential Regression """

    swap_vec = mdl.calc_swap(r0_vec, t_swap_fixings, delta, strike, notional)
    X_train, y_train, z_train = training_data(
        r0_vec=r0_vec, t0=0.0, calc_dU_dr=calc_dswap_dr, calc_dPrd_dr=calc_dswpt_dr, use_av=use_av
    )

    X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

    diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

    X_test_scaled, _, _ = scalar.transform(X_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    RMSE_price = torch.sqrt(torch.mean((y_pred - y_test) ** 2))
    MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_test))

    """ Plot results """
    fig, ax = plt.subplots(2, sharex='col')
    # Plot price function
    ax[0].plot(X_train.flatten(), y_train.flatten(), 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[0].plot(X_test.flatten(), y_pred, label='DiffReg', color='orange')
    ax[0].plot(X_test, y_test, color='black', label='MC (Bump and reval)')
    ax[0].set_ylabel('Price')
    ax[0].text(0.05, 0.8, f'RMSE = {RMSE_price:.2f}', fontsize=8, transform=ax[0].transAxes)

    # Plot delta function
    ax[1].plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[1].plot(X_test, z_pred, label='DiffReg', color='orange')
    ax[1].plot(X_test[1:], z_test, color='black', label='MC (Bump and reval)')
    ax[1].set_xlabel('Swap(0)')
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

    #plt.savefig(get_plot_path('vasicek_AAD_DiffReg_BerPayerSwpt.png'), dpi=400)
    plt.show()