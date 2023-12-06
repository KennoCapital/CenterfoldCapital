import torch
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import BarrierPayerSwaption
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSim, RNG
from application.utils.path_config import get_plot_path, get_data_path
from application.experiments.vasicek.vasicek_hedge_tools import training_data

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

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=True)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'terminal'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Plot snapshots of different maturities
    fig, ax = plt.subplots(4, 2)

    T = [1.0, 0.5, 0.25, 0.05]
    file_str = ['1', '05', '025', '005']
    if len(T) != 4 & len(file_str) != 4:
        raise ValueError('Maturities and file_str must contain exactly 4 elements!')

    for i in range(len(T)):
        exerciseDate = torch.tensor(T[i])
        delta = torch.tensor(0.25)
        swapFirstFixingDate = exerciseDate
        swapLastFixingDate = exerciseDate + torch.tensor(5.0)
        notional = torch.tensor(1e6)

        t_swap_fixings = torch.linspace(
            float(swapFirstFixingDate),
            float(swapLastFixingDate),
            int((swapLastFixingDate - swapFirstFixingDate) / delta + 1)
        )

        strike = torch.tensor(0.0871)  #mdl.calc_swap_rate(r0, t_swap_fixings, delta)

        hedge_times = 100
        dTL = torch.linspace(0.0, float(swapFirstFixingDate), hedge_times + 1)
        prd = BarrierPayerSwaption(
            strike=strike,
            exerciseDate=exerciseDate,
            delta=delta,
            barrier=torch.tensor(20000),
            obsTL=dTL,
            swapFirstFixingDate=swapFirstFixingDate,
            swapLastFixingDate=swapLastFixingDate,
            notional=notional,
            smooth=torch.tensor(0.05),
            smoothing='sigmoid'
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
                cPrd = BarrierPayerSwaption(
                        strike=strike,
                        exerciseDate=exerciseDate - t0,
                        delta=delta,
                        barrier=prd.barrier,
                        obsTL=dTL[int(t0):],
                        swapFirstFixingDate=swapFirstFixingDate - t0,
                        swapLastFixingDate=swapLastFixingDate - t0,
                        notional=notional,
                        smooth=prd.smooth,
                        smoothing=prd.smoothing
                )
                cRng = RNG(seed=seed, use_av=use_av)
                payoffs = mcSim(cPrd, cMdl, cRng, len(r0_vec))
                return payoffs

            ones = torch.ones_like(r0_vec)
            res = jvp(_payoffs, r0_vec, ones, create_graph=False)
            return res

        """ Calculate `true` swaption price using Monte Carlo for comparison """


        file_path = get_data_path(f'vasicek_Barrier_test_set_T_{file_str[i]}.pkl')

        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                X_test, y_mdl, z_mdl = pickle.load(file)
        else:
            r0_test_vec = torch.linspace(r0_min, r0_max, N_test)
            y_mdl = torch.full_like(r0_test_vec, torch.nan)
            for j in tqdm(range(len(r0_test_vec))):
                tmp_mdl = Vasicek(a, b, sigma, r0_test_vec[j], use_ATS=True, use_euler=False, measure='terminal')
                tmp_rng = RNG(seed=seed, use_av=True)
                y_mdl[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 200000)))
            y_mdl = y_mdl.reshape(-1, 1)

            X_test = tmp_mdl.calc_swap(r0_test_vec, t_swap_fixings, delta, strike, notional).reshape(-1, 1)
            z_mdl = y_mdl.diff(dim=0) / X_test.diff(dim=0)

            with open(file_path, 'wb') as file:
                pickle.dump(tuple([X_test, y_mdl, z_mdl]), file, pickle.HIGHEST_PROTOCOL)


        """ Estimate Price and Delta using Differential Regression """

        X_train, y_train, z_train = training_data(r0_vec = r0_vec, t0 = 0.0, calc_dU_dr = calc_dswap_dr, calc_dPrd_dr = calc_dswpt_dr, use_av = True)

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

        X_test_scaled, _, _ = scalar.transform(X_test, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        RMSE_price = torch.sqrt(torch.mean((y_pred - y_mdl) ** 2))
        MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_mdl))

        """ Plot results"""

        if i < 2:
            row = 0
        else:
            row = 2

        if i % 2 == 0:
            col = 0
        else:
            col = 1

        # Plot price function
        ax[row, col].plot(X_train.flatten(), y_train.flatten(), 'o', color='gray', alpha=0.25, label='Pathwise samples')
        ax[row, col].plot(X_test.flatten(), y_pred, label='DiffReg', color='orange')
        ax[row, col].plot(X_test, y_mdl, color='black', label='MC (Bump and reval)')
        ax[row, col].set_ylabel('Price')
        ax[row, col].text(0.05, 0.8, f'RMSE = {RMSE_price:.2f}', fontsize=8, transform=ax[row, col].transAxes)
        ax[row, col].set_title(f'T={T[i]}')

        # Plot delta function
        ax[row + 1, col].plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Pathwise samples')
        ax[row + 1, col].plot(X_test, z_pred, label='DiffReg', color='orange')
        ax[row + 1, col].plot(X_test[1:], z_mdl, color='black', label='MC (Bump and reval)')
        ax[row + 1, col].set_xlabel('Swap(0)')
        ax[row + 1, col].set_ylabel('Delta')
        ax[row + 1, col].set_ylim(-1.2, 1.2)
        ax[row + 1, col].text(0.05, 0.8, f'MAE = {MAE_delta:.4f}', fontsize=8, transform=ax[row + 1, col].transAxes)

        # Adjust size of subplots
        box0 = ax[row, col].get_position()
        ax[row, col].set_position([box0.x0, box0.y0 - box0.height * 0.1, box0.width, box0.height * 0.9])

        box1 = ax[row + 1, col].get_position()
        ax[row + 1, col].set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])


    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), draggable=True, ncol=3, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    # Title
    av_str = 'with AV' if use_av else 'without AV'
    fig.suptitle(prd.name + f'\nalpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str)

    #plt.savefig(get_plot_path('vasicek_AAD_DiffReg_EuSwpt_closetoT_Naive.png'), dpi=400)
    plt.show()
