import os
import torch
import pickle
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker
from tqdm import tqdm
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import PortfolioEuropeanPayerSwaption
from application.engine.mcBase import mcSim, RNG
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.utils.path_config import get_data_path, get_plot_path
from application.experiments.vasicek.vasicek_hedge_tools import training_data
from application.utils.prd_name_conventions import float_to_time_str

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    name = 'Call Spread'
    file_path = get_data_path('vasicek_call_spread.pkl')

    seed = 1234
    N_train = 256
    N_test = 170
    use_av = True

    r0_min = -0.02
    r0_max = 0.15

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    use_SVD = True
    bias = True
    include_interactions = False
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=use_SVD,
                                               bias=bias, include_interactions=include_interactions)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(
        0.25)  # TODO: Carefull results are strange if a is high and the underlying swaps have long maturity
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDates = torch.tensor([1.0])
    fixRates = [torch.tensor([0.06, 0.09])]
    deltas = [torch.tensor([0.25, 0.25])]
    swapFirstFixingDates = [torch.tensor([1.0, 1.0])]
    swapLastFixingDates = [torch.tensor([6.0, 6.0])]
    notionals = [torch.tensor([1e6, 1e6])]
    weights = [torch.tensor([1.0, -1.0])]

    prd = PortfolioEuropeanPayerSwaption(
        exerciseDates=exerciseDates,
        fixRates=fixRates,
        deltas=deltas,
        swapFirstFixingDates=swapFirstFixingDates,
        swapLastFixingDates=swapLastFixingDates,
        notionals=notionals,
        weights=weights
    )

    """ Helper functions for generating training data of pathwise payoffs and deltas """


    def calc_dswap_dr(r0_vec, t0):
        ones = torch.ones_like(r0_vec)
        res = []
        for i, fixings in enumerate(prd.swapFixingDates):
            for j in range(len(prd.fixRates[i])):
                def _swap_price(r0_vec):
                    swap = mdl.calc_swap(r0_vec,
                                         prd.swapFixingDates[i][j] - t0,
                                         prd.deltas[i][j],
                                         prd.fixRates[i][j],
                                         notionals[i][j])
                    return swap



                Jv = jvp(_swap_price, r0_vec, ones, create_graph=False)
                res.append(Jv)
        prices = torch.vstack([x[0] for x in res])
        derivs = torch.vstack([x[1] for x in res])
        return prices.T, derivs.T

    def calc_dswap_rate_dr(r0_vec, t0):
        ones = torch.ones_like(r0_vec)
        res = []
        for i, fixings in enumerate(prd.swapFixingDates):
            for j in range(len(prd.fixRates[i])):
                def _swap_rate(r0_vec):
                    swap_rate = mdl.calc_swap_rate(r0_vec,
                                                   prd.swapFixingDates[i][j] - t0,
                                                   prd.deltas[i][j])
                    return swap_rate

                Jv = jvp(_swap_rate, r0_vec, ones, create_graph=False)
                res.append(Jv)
        rates = torch.vstack([x[0] for x in res])
        derivs = torch.vstack([x[1] for x in res])
        return rates.T, derivs.T

    def calc_dcashflow_dr(r0_vec, t0):
        def _payoff(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='risk_neutral')
            cPrd = PortfolioEuropeanPayerSwaption(
                exerciseDates=exerciseDates - t0,
                fixRates=fixRates,
                deltas=deltas,
                swapFirstFixingDates=[d - t0 for d in swapFirstFixingDates],
                swapLastFixingDates=[d - t0 for d in swapLastFixingDates],
                notionals=notionals,
                weights=weights
            )
            cRng = RNG(seed=seed, use_av=use_av)
            cashflow = mcSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec))
            return cashflow

        ones = torch.ones_like(r0_vec)
        cashflows, derivs = jvp(func=_payoff, inputs=r0_vec, v=ones, create_graph=False)
        return cashflows.T, derivs.T

    def calc_dpayoff_dr(r0_vec, t0):
        def _payoff(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='risk_neutral')
            cPrd = PortfolioEuropeanPayerSwaption(
                exerciseDates=exerciseDates - t0,
                fixRates=fixRates,
                deltas=deltas,
                swapFirstFixingDates=[d - t0 for d in swapFirstFixingDates],
                swapLastFixingDates=[d - t0 for d in swapLastFixingDates],
                notionals=notionals,
                weights=weights
            )
            cRng = RNG(seed=seed, use_av=use_av)
            cashflow = mcSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec))
            payoffs = torch.sum(cashflow, dim=0)
            return payoffs

        ones = torch.ones_like(r0_vec)
        payoffs, derivs = jvp(func=_payoff, inputs=r0_vec, v=ones, create_graph=False)
        return payoffs.reshape(-1, 1), derivs.reshape(-1, 1)


    """ Calculate `true` swaption price using Monte Carlo for comparison """

    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            x_test, y_test, z_test, swap_rate = pickle.load(file)
    else:
        # State variables
        r0_test_vec = torch.linspace(float(r0_min), float(r0_max), int(N_test) + 1)

        # Market observables (underlying swaps and swap rates)
        x_test = [torch.hstack([
            mdl.calc_swap(r0_test_vec, prd.swapFixingDates[i][j], deltas[i][j], fixRates[i][j],
                          notionals[i][j]).reshape(-1, 1)
            for j in range(len(prd.fixRates[i]))
        ]) for i in range(prd.numT)]
        swap_rate_test = [mdl.calc_swap_rate(r0_test_vec, prd.swapFixingDates[i][j], deltas[i][j]).reshape(-1, 1)
                          for i in range(prd.numT) for j in range(len(prd.fixRates[i]))]

        # Payments (cashflows) and payoffs
        cf_test = torch.full(size=(N_test + 1, prd.numTrades), fill_value=torch.nan)
        y_test = torch.full_like(r0_test_vec, torch.nan)
        for i, r in tqdm(enumerate(r0_test_vec), desc='Calculating test prices with MC', total=len(r0_test_vec)):
            cMdl = Vasicek(a, b, sigma, r, use_ATS=True, use_euler=False, measure='risk_neutral')
            cRng = RNG(use_av=use_av, seed=seed)
            payments = mcSim(prd, cMdl, cRng, N=50000)
            payoff = torch.sum(payments, dim=0)
            cf_test[i] = torch.mean(payments, dim=1)
            y_test[i] = torch.mean(payoff)

        # Reshape into a dataset
        x_test = torch.hstack(x_test)
        swap_rate_test = torch.hstack(swap_rate_test)
        y_test = y_test.reshape(-1, 1)

        # Pathwise differentials of payoffs
        solve_rowwise = lambda dxdr_, dydr_: (torch.pinverse(dxdr_.T) @ dydr_.T).flatten()
        dxdr = x_test.diff(dim=0)
        dydr = y_test.diff(dim=0)
        equations = (
            (dxdr[i, :].reshape(-1, 1), dydr[i, :].reshape(-1, 1)) for i in range(len(r0_test_vec) - 1)
        )
        solutions = itertools.starmap(solve_rowwise, equations)
        z_test = torch.vstack(list(solutions))

        cf_delta_test = cf_test.diff(dim=0) / x_test.diff(dim=0)

        # with open(file_path, 'wb') as file:
        #    pickle.dump(tuple([x_test, y_test, z_test, swap_rate_test]), file, pickle.HIGHEST_PROTOCOL)


    """ Estimate Price and Delta using Differential Regression """


    # TODO
    ## Estimator for (price of port, swaps)
    ## Estimator for (delta of port, swaps)
    ## Estimator for (price of port, swap-rate)
    ## Estimator for (delta of port, swap-rate)
    ## Estimator for (price of trades, swaps)
    ## Estimator for (delta of trades, swaps)
    ## Estimator for (price of trades, swap-rate)
    ## Estimator for (delta of trades, swap-rate)

    def diff_reg_fit_predict(x_train, y_train, z_train, x_test):
        x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)
        diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)
        x_test_scaled, _, _ = scalar.transform(x_test, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(x_test_scaled, predict_derivs=True)
        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        return y_pred, z_pred

    """ Value and sensitivity wrt. underlying swap of each swaption """
    x_train, y_train, z_train = training_data(r0_vec, 0.0, calc_dswap_dr, calc_dcashflow_dr, use_av)
    ys = []
    zs = []
    for i in range(prd.numTrades):
        y_pred, z_pred = diff_reg_fit_predict(
            x_train[:, i].reshape(-1, 1),
            y_train[:, i].reshape(-1, 1),
            z_train[:, i].reshape(-1, 1),
            x_test[:, i].reshape(-1, 1)
        )
        ys.append(y_pred)
        zs.append(z_pred)
    y_pred = torch.hstack(ys)
    z_pred = torch.hstack(zs)

    RMSE_value = torch.sqrt(torch.mean(torch.pow(y_pred - cf_test, 2.0), dim=0))
    MAE_delta = torch.mean(torch.abs(z_pred[1:] - cf_delta_test), dim=0)

    fig, ax = plt.subplots(nrows=2, ncols=prd.numTrades, sharex='col', sharey='row', figsize=(10, 6))
    for i in range(prd.numTrades):
        ax[0, i].plot(x_train[:, i], y_train[:, i], 'o', color='gray', alpha=0.25, label='Pathwise Samples')
        ax[0, i].plot(x_test[:, i], y_pred[:, i], color='orange', label='DiffReg')
        ax[0, i].plot(x_test[:, i], cf_test[:, i], color='black', label='MC (Bump and reval)')

        ax[1, i].plot(x_train[:, i], z_train[:, i], 'o', color='gray', alpha=0.25)
        ax[1, i].plot(x_test[:, i], z_pred[:, i], color='orange')
        ax[1, i].plot(x_test[1:, i], cf_delta_test[:, i], color='black')

        ax[0, i].axvline(x=0.0, color='gray', ls='--', alpha=0.25)
        ax[1, i].axvline(x=0.0, color='gray', ls='--', alpha=0.25)

        long_short_flag = 'Long' if prd.weights[0][i] >= 0.0 else 'Short'
        ax[0, i].set_title(f'{long_short_flag} {weights[0][i]:.2f}')

        ax[1, i].set_xlabel(f'swap(0, '
                            f'{float_to_time_str(prd.swapFirstFixingDates[0][i])}, '
                            f'{float_to_time_str(prd.swapLastFixingDates[0][i])}, '
                            f'{float_to_time_str(prd.deltas[0][i])}) @ {prd.fixRates[0][i] * 100:.2f}%')

        ax[0, i].xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
        ax[1, i].xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
        ax[0, i].yaxis.set_major_formatter(matplotlib.ticker.EngFormatter())
        ax[1, i].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1e5))

        ax[0, i].text(0.05, 0.8, f'RMSE = {RMSE_value[i]:.2f}', fontsize=8, transform=ax[0, i].transAxes)
        ax[1, i].text(0.05, 0.8, f'MAE = {MAE_delta[i]:.3f}', fontsize=8, transform=ax[1, i].transAxes)

    ax[0, 0].set_ylabel('Price')
    ax[1, 0].set_ylabel('Delta')

    fig.suptitle(f'Decomposition of {name}')
    plt.show()
