import torch
import matplotlib.pyplot as plt
import itertools
import pickle
import os
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
    file_path = get_data_path('vasicek_Basket_test_set.pkl')

    seed = 1234
    N_train = 1024
    use_av = True

    test_bump_size_bp = torch.tensor(1.0)

    r0_min = torch.tensor(-0.02)
    r0_max = torch.tensor(0.15)

    N_test = torch.round((r0_max - r0_min) * test_bump_size_bp * 10000)

    r0_vec = torch.linspace(float(r0_min), float(r0_max), N_train)

    # Setup Differential Regressor, and Scalar
    deg = 5
    alpha = 1.0
    use_SVD = True
    bias = True
    include_interactions = False

    diff_reg = DifferentialPolynomialRegressor(
        deg=deg,
        alpha=alpha,
        use_SVD=use_SVD,
        bias=bias,
        include_interactions=include_interactions
    )
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
    numTenors = 3
    numMoneyness = 5
    n = numTenors * numMoneyness

    delta = torch.tensor([0.25])
    exerciseDate = torch.tensor([1.0])
    deltas = torch.full(size=(n, ), fill_value=float(delta))
    swapLastFixingDates = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0,
                                        5.0, 5.0, 5.0, 5.0, 5.0,
                                        10.0, 10.0, 10.0, 10.0, 10.0])
    swapFirstFixingDates = torch.full(size=(n, ), fill_value=float(exerciseDate))
    notional = torch.tensor(1e6)

    weights = torch.ones(size=(n,))
    notionals = torch.full(size=(n, ), fill_value=float(notional))

    t0_swap_fixings = [torch.linspace(
        float(exerciseDate),
        float(swapLastFixingDates[i] + exerciseDate),
        int((swapLastFixingDates[i]) / delta + 1)
    ) for i in range(n)]

    moneyness = torch.tensor([0.50, 0.75, 1.0, 1.25, 1.50])
    strike = torch.tensor([0.0])

    fixRates = torch.full_like(swapLastFixingDates, torch.nan)
    for i in range(numTenors):
        for j, m in enumerate(moneyness):
            idx = i * numMoneyness + j
            fixRates[idx] = mdl.calc_swap_rate(r0, t0_swap_fixings[idx], torch.tensor(delta)) * m

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
        for i in range(n):
            def _swap_price(r0_vec):
                tau = t0_swap_fixings[i] - t0
                S = mdl.calc_swap(r0_vec, tau, deltas[i], fixRates[i], notionals[i])
                return S
            Jv = jvp(_swap_price, r0_vec, ones, create_graph=False)
            res.append(Jv)
        prices = torch.vstack([x[0] for x in res]).T
        derivs = torch.vstack([x[1] for x in res]).T
        return prices, derivs

    def calc_dbasket_dr(r0_vec, t0):
        def _basket(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = BasketEuropeanPayerSwaptions(
                    exerciseDate=exerciseDate - t0,
                    fixRates=fixRates,
                    deltas=deltas,
                    swapFirstFixingDates=swapFirstFixingDates - t0,
                    swapLastFixingDates=swapLastFixingDates - t0,
                    weights=weights,
                    strike=strike,
                    notionals=notionals
                )
            cRng = RNG(seed=seed, use_av=use_av)
            payoff = mcSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec))
            return payoff
        ones = torch.ones_like(r0_vec)
        prices, derivs = jvp(func=_basket, inputs=r0_vec, v=ones, create_graph=False)
        return prices, derivs


    """ Calculate `true` swaption price using Monte Carlo for comparison """

    if os.path.isfile(file_path):
        with open(file_path, 'rb') as file:
            x_test, y_test, z_test = pickle.load(file)
    else:
        r0_test_vec = torch.linspace(float(r0_min), float(r0_max), int(N_test) + 1)
        x_test = torch.hstack(
            [mdl.calc_swap(r0_test_vec, t0_swap_fixings[i], deltas[i], fixRates[i], notionals[i]).reshape(-1, 1)
             for i in range(n)]
        )
        y_test = torch.full_like(r0_test_vec, torch.nan)
        for i, r in tqdm(enumerate(r0_test_vec), desc='Calculating test prices with MC', total=len(r0_test_vec)):
            cMdl = Vasicek(a, b, sigma, r, use_ATS=True, use_euler=False, measure='terminal')
            cRng = RNG(use_av=use_av, seed=seed)
            payoff = mcSim(prd, cMdl, cRng, N=50000)
            y_test[i] = torch.mean(payoff)

        x_test = x_test.reshape(-1, n)
        y_test = y_test.reshape(-1, 1)

        solve_rowwise = lambda dxdr_, dydr_: (torch.pinverse(dxdr_.T) @ dydr_.T).flatten()
        dxdr = x_test.diff(dim=0)
        dydr = y_test.diff(dim=0)
        equations = (
            (dxdr[i, :].reshape(-1, 1), dydr[i, :].reshape(-1, 1)) for i in range(len(r0_test_vec) - 1)
        )
        solutions = itertools.starmap(solve_rowwise, equations)
        z_test = torch.vstack(list(solutions))

        with open(file_path, 'wb') as file:
            pickle.dump(tuple([x_test, y_test, z_test]), file, pickle.HIGHEST_PROTOCOL)

    """ Estimate Price and Delta using Differential Regression """

    x_train, y_train, z_train = training_data(
        r0_vec=r0_vec, t0=0.0, calc_dU_dr=calc_dswap_dr, calc_dPrd_dr=calc_dbasket_dr, use_av=use_av
    )

    x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)

    diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)

    x_test_scaled, _, _ = scalar.transform(x_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(x_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    """ Plot results """
    RMSE_price = torch.sqrt(torch.mean((y_pred - y_test) ** 2))
    MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_test), dim=0)

    dyn_alpha = min(0.25 * 1024 / N_train, 0.5)
    av_str = 'with AV' if use_av else 'without AV'
    interactions_str = 'with interactions' if include_interactions else 'without interactions'

    # Price
    fig, ax = plt.subplots(nrows=numMoneyness, ncols=numTenors, sharex='all', sharey='all')
    for i, m in enumerate(moneyness):
        for j in range(numTenors):
            ax[i, j].plot(x_train[:, i].flatten(), y_train.flatten(), 'o', color='gray', alpha=dyn_alpha, label='Sample Payoffs')
            ax[i, j].plot(x_test[:, i].flatten(), y_pred, 'o', color='orange', alpha=0.5, label='DiffReg')
            ax[i, j].plot(x_test[:, i].flatten(), y_test, color='black', label='MC (bump & reval)')
    plt.show()

    # Delta
    fig, ax = plt.subplots(nrows=numMoneyness, ncols=numTenors, sharex='all', sharey='all')
    for i, m in enumerate(moneyness):
        for j in range(numTenors):
            ax[i, j].plot(x_train[:, i].flatten(), z_train[:, i].flatten(), 'o', color='gray', alpha=dyn_alpha, label='Sample Differentials')
            ax[i, j].plot(x_test[1:, i].flatten(), z_pred[1:, i].flatten(), 'o', color='orange', alpha=0.5, label='DiffReg')
            ax[i, j].plot(x_test[1:, i].flatten(), z_test[:, i].flatten(), color='black', label='MC (bump & reval)')
    plt.show()


