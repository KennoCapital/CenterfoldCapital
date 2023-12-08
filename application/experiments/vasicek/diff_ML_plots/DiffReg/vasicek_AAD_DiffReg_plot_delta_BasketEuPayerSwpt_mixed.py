import os
import torch
import matplotlib.ticker
import matplotlib.pyplot as plt
import itertools
import pickle
from tqdm import tqdm
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import BasketEuropeanPayerSwaptions
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import RNG, mcSim
from application.experiments.vasicek.vasicek_hedge_tools import training_data
from application.utils.path_config import get_plot_path, get_data_path

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024
    use_av = True

    nList = [3, 7, 15, 25, 50]
    alphaList = [0.0, 1.0]

    test_bump_size_bp = torch.tensor(1.0)

    r0_min = torch.tensor(-0.02)
    r0_max = torch.tensor(0.15)

    N_test = torch.round((r0_max - r0_min) * test_bump_size_bp * 10000)

    r0_vec = torch.linspace(float(r0_min), float(r0_max), N_train)

    # Setup Differential Regressor, and Scalar
    deg = 9
    use_SVD = True
    bias = True
    include_interactions = False
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
    delta = torch.tensor([0.25])
    notional = torch.tensor(1e6)
    exerciseDate = torch.tensor([1.0])

    # Experiment
    maturityFloatList = torch.tensor([
        0.25, 0.5, 1.0, 2.0, 5.0
        #7.0, 10.0, 20.0, 30.0
    ])

    fig, ax = plt.subplots(nrows=len(nList), ncols=len(alphaList), sharey='all', sharex='row', figsize=(10, 12))
    for k, n_underlying in enumerate(nList):

        rngExp = torch.Generator().manual_seed(420)
        weights = torch.rand(size=(n_underlying,), generator=rngExp)
        weights = weights / torch.sum(torch.abs(weights))
        mats = maturityFloatList[torch.randint(size=(n_underlying,), low=0, high=len(maturityFloatList), generator=rngExp)]
        fixRates = torch.normal(size=(n_underlying,), mean=float(0.06), std=0.02, generator=rngExp)

        swapLastFixingDates = mats + exerciseDate

        deltas = torch.full(size=(n_underlying, ), fill_value=float(delta))
        swapFirstFixingDates = torch.full(size=(n_underlying, ), fill_value=float(exerciseDate))
        notionals = torch.full(size=(n_underlying, ), fill_value=float(notional))

        t0_swap_fixings = [torch.linspace(
            float(exerciseDate),
            float(swapLastFixingDates[i]),
            int((swapLastFixingDates[i] - exerciseDate) / deltas[i] + 1)
        ) for i in range(n_underlying)]

        swapValues = torch.full_like(swapLastFixingDates, torch.nan)

        for i in range(n_underlying):
            swapValues[i] = mdl.calc_swap(r0, t0_swap_fixings[i], deltas[i], fixRates[i], notionals[i])

        strike = torch.dot(swapValues, weights)

        prd = BasketEuropeanPayerSwaptions(
            exerciseDate=exerciseDate,
            fixRates=fixRates,
            deltas=deltas,
            swapFirstFixingDates=swapFirstFixingDates,
            swapLastFixingDates=swapLastFixingDates,
            weights=weights,
            strike=strike,
            notionals=notionals
        )

        """ Helper functions for generating training data of pathwise payoffs and deltas """

        def calc_dswap_dr(r0_vec, t0):
            ones = torch.ones_like(r0_vec)
            res = []
            for i in range(n_underlying):
                def _swap_price(r0_vec):
                    S = mdl.calc_swap(r0_vec, prd.swapFixingDates[i] - t0, deltas[i], fixRates[i], notionals[i])
                    return S
                Jv = jvp(_swap_price, r0_vec, ones, create_graph=False)
                res.append(Jv)
            prices = torch.vstack([x[0] for x in res]).T
            derivs = torch.vstack([x[1] for x in res]).T
            return prices, derivs

        def calc_dbasket_dr(r0_vec, t0):
            def _basket(r0_vec):
                cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure=measure)
                cPrd = BasketEuropeanPayerSwaptions(
                        exerciseDate=prd.exerciseDate - t0,
                        fixRates=prd.fixRates,
                        deltas=prd.deltas,
                        swapFirstFixingDates=prd.swapFirstFixingDates - t0,
                        swapLastFixingDates=prd.swapLastFixingDates - t0,
                        weights=prd.weights,
                        strike=prd.strike,
                        notionals=prd.notionals
                    )
                cRng = RNG(seed=seed, use_av=use_av)
                payoff = mcSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec))
                return payoff
            ones = torch.ones_like(r0_vec)
            prices, derivs = jvp(func=_basket, inputs=r0_vec, v=ones, create_graph=False)
            return prices, derivs


        """ Calculate `true` swaption price using Monte Carlo for comparison """

        file_path = get_data_path(f'vasicek_Basket_test_set_{n_underlying}.pkl')

        if os.path.isfile(file_path):
            with open(file_path, 'rb') as file:
                x_test, y_test, z_test = pickle.load(file)
        else:
            r0_test_vec = torch.linspace(float(r0_min), float(r0_max), int(N_test) + 1)
            x_test = torch.hstack([
                mdl.calc_swap(r0_test_vec, prd.swapFixingDates[i], prd.deltas[i], prd.fixRates[i], prd.notionals[i]).reshape(-1, 1)
                for i in range(prd.n)
            ])
            y_test = torch.full_like(r0_test_vec, torch.nan)
            for i, r in tqdm(enumerate(r0_test_vec),
                             desc=f'Calculating test prices with MC ({prd.n} underlying)',
                             total=len(r0_test_vec)):
                cMdl = Vasicek(a, b, sigma, r, use_ATS=True, use_euler=False, measure=measure)
                cRng = RNG(use_av=use_av, seed=seed)
                payoff = mcSim(prd, cMdl, cRng, N=50000)
                y_test[i] = torch.mean(payoff)

            x_test = x_test.reshape(-1, prd.n)
            y_test = y_test.reshape(-1, 1)

            solve_rowwise = lambda dxdr_, dydr_: (torch.pinverse(dxdr_.T) @ dydr_.T).reshape(-1, prd.n).sum(dim=0)
            dxdr = x_test.diff(dim=0)
            dydr = y_test.diff(dim=0)
            equations = (
                (dxdr[i, :].reshape(-1, 1), dydr[i, :].reshape(-1, 1)) for i in range(len(r0_test_vec) - 1)
            )
            solutions = itertools.starmap(solve_rowwise, equations)
            z_test = torch.vstack(list(solutions))

            with open(file_path, 'wb') as file:
                pickle.dump(tuple([x_test, y_test, z_test]), file, pickle.HIGHEST_PROTOCOL)

        # Value of basket at time 0
        b_test = x_test @ weights

        """ Estimate Price and Delta using Differential Regression """

        for j, alpha in enumerate(alphaList):
            diff_reg = DifferentialPolynomialRegressor(
                deg=deg,
                alpha=alpha,
                use_SVD=use_SVD,
                bias=bias,
                include_interactions=include_interactions
            )

            def diff_reg_fit_predict(x_train, y_train, z_train, x_test):
                x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)
                diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)
                x_test_scaled, _, _ = scalar.transform(x_test, None, None)
                y_pred_scaled, z_pred_scaled = diff_reg.predict(x_test_scaled, predict_derivs=True)
                _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

                return y_pred, z_pred

            x_train, y_train, z_train = training_data(
                r0_vec=r0_vec, t0=0.0, calc_dU_dr=calc_dswap_dr, calc_dPrd_dr=calc_dbasket_dr, use_av=use_av
            )

            y_pred, z_pred = diff_reg_fit_predict(x_train, y_train, z_train, x_test)

            b_train = x_train @ weights

            """ Plot results """
            RMSE_value = torch.sqrt(torch.mean((y_pred - y_test) ** 2))
            MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_test))

            ax[k, j].plot(b_train / strike, y_train, 'o', color='gray', alpha=0.25, label='Pathwise Samples')
            ax[k, j].plot(b_test / strike, y_pred, color='orange', label='DiffReg')
            ax[k, j].plot(b_test / strike, y_test, color='black', label='Monte Carlo')

            ax[k, j].yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

            ax[k, j].text(0.05, 0.8, f'RMSE (value) = {RMSE_value:.2f}', fontsize=8, transform=ax[k, j].transAxes)
            ax[k, j].text(0.05, 0.7, f'MAE (delta) = {MAE_delta:.4f}', fontsize=8, transform=ax[k, j].transAxes)

            if j == 0:
                ax[k, j].set_ylabel(f'{n_underlying} Underlying')

            if k == 0:
                ax[k, j].set_title(f'$\\alpha={alpha}$')

            if k == len(nList) - 1:
                ax[k, j].set_xlabel('Moneyness $\\left(\\frac{Basket}{K_{Basket}}\\right)$')

            box = ax[k, j].get_position()
            ax[k, j].set_position([box.x0, box.y0, box.width, box.height * 0.85])


    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.95))

    fig.suptitle(f'European Swaption Basket')
    plt.savefig(get_plot_path('07_vasicek_ADD_DiffReg_value_BasketEuPayerSwaption_mixed.png'))
    plt.show()
