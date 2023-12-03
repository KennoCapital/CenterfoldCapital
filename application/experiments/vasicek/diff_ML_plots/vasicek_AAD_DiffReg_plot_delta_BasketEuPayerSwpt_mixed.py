import torch
import matplotlib.ticker
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
from application.utils.prd_name_conventions import float_to_time_str

"""
    Assume that a client wants to bet that 1Y from now,
    1) the short rates (3M,  6M,  1Y,  2Y) are going to be higher than their current level,
    2) the long  rates (5Y, 10Y, 20Y, 30Y) are going to be lower than their current level.
    
    A possible long-short strategy for this is to consider a portfolio that is
    - LONG  swaps with high fixing-rates and short maturities,
    - SHORT swaps with low  fixing-rates and short maturities,
    - LONG  swaps with low  fixing-rates and long  maturities,
    - SHORT swaps with high fixing-rates and long  maturities.
    
    To make the this sort of trade and still have down-side protection in the form of limited losses
    the client wants to make the bet in the form of Basket European Payer Swaption contract, which has the
    above mentioned portfolio as its underlying.  
    
    We consider 
    *   0.75 * swap_rate as low fixing-rates,
    *   1.25 * swap_rate as high fixings-rates.
    
    The strike of the Basket Swaption is set equal to the value of the underlying basket, i.e. at-the-money (ATM).   
"""

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    file_path = get_data_path('vasicek_Basket_mixed_test_set.pkl')

    seed = 1234
    N_train = 1024
    use_av = True

    test_bump_size_bp = torch.tensor(1.0)

    r0_min = torch.tensor(-0.02)
    r0_max = torch.tensor(0.15)

    N_test = torch.round((r0_max - r0_min) * test_bump_size_bp * 10000)

    r0_vec = torch.linspace(float(r0_min), float(r0_max), N_train)

    # Experiment
    maturityFloatList = [
        0.25, 0.5, 1.0, 2.0,    # short maturities
        5.0, 10.0, 20.0, 30.0   # long  maturities
    ]
    moneynessPctList = [0.75, 1.25]
    numMat = len(maturityFloatList)
    numMoneyness = len(moneynessPctList)

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
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    delta = torch.tensor([0.25])
    notional = torch.tensor(1e6)
    exerciseDate = torch.tensor([1.0])

    weights = torch.tensor([
        1.0, 1.0, 1.0, 1.0,         # Low  fix rate, short maturity
        -1.0, -1.0, -1.0, -1.0,     # Low  fix rate, long  maturity
        -1.0, -1.0, -1.0, -1.0,     # High fix rate, short maturity
        1.0, 1.0, 1.0, 1.0,         # High fix rate, long  maturity
    ])

    swapLastFixingDates = torch.tensor(
        maturityFloatList + maturityFloatList
    ) + exerciseDate

    n = len(weights)

    deltas = torch.full(size=(n, ), fill_value=float(delta))
    swapFirstFixingDates = torch.full(size=(n, ), fill_value=float(exerciseDate))
    notionals = torch.full(size=(n, ), fill_value=float(notional))

    t0_swap_fixings = [torch.linspace(
        float(exerciseDate),
        float(swapLastFixingDates[i]),
        int((swapLastFixingDates[i] - exerciseDate) / deltas[i] + 1)
    ) for i in range(n)]

    swapValues = torch.full_like(swapLastFixingDates, torch.nan)
    fixRates = torch.full_like(swapLastFixingDates, torch.nan)
    for i, m in enumerate(moneynessPctList):
        for j, T in enumerate(maturityFloatList):
            idx = i * numMat + j
            fixRates[idx] = mdl.calc_swap_rate(r0, t0_swap_fixings[idx], torch.tensor(delta)) * m
            swapValues[idx] = mdl.calc_swap(r0, t0_swap_fixings[idx], deltas[idx], fixRates[idx], notionals[idx])

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

    backgroundcolor = {1.0: 'lightgreen', -1.0: 'pink'}
    fixRateLabel = {0.75: 'Low fix rate', 1.25: 'High fix rate'}

    # Price
    fig, ax = plt.subplots(nrows=numMoneyness, ncols=numMat, sharey='all', figsize=(12, 8))
    for i, m in enumerate(moneynessPctList):
        for j in range(numMat):
            idx = i * numMat + j
            ax[i, j].plot(x_train[:, idx].flatten(), y_train.flatten(), 'o', color='gray', alpha=dyn_alpha, label='Sample Payoffs')
            ax[i, j].plot(x_test[:, idx].flatten(), y_pred, 'o', color='orange', alpha=0.5, label='DiffReg')
            ax[i, j].plot(x_test[:, idx].flatten(), y_test, color='black', label='MC (bump & reval)')
            ax[i, j].set_facecolor(backgroundcolor[float(weights[idx])])

            if j == 0:
                ax[i, j].set_ylabel(fixRateLabel[float(moneynessPctList[i])])

            if i == 0:
                ax[i, j].set_title(float_to_time_str(maturityFloatList[j]))

            # Adjust size of price plot
            box = ax[i, j].get_position()
            ax[i, j].set_position([box.x0, box.y0, box.width, box.height * 0.8])

            ax[i, j].xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.95))

    fig.suptitle('Basket Value')
    plt.savefig(get_plot_path('07_vasicek_ADD_DiffReg_value_BasketEuPayerSwaption_mixed.png'), dpi=400)
    plt.show()

    # Delta
    fig, ax = plt.subplots(nrows=numMoneyness, ncols=numMat, sharey='all', figsize=(12, 8))
    for i, m in enumerate(moneynessPctList):
        for j in range(numMat):
            idx = i * numMat + j
            ax[i, j].plot(x_train[:, idx].flatten(), z_train[:, idx].flatten(), 'o', color='gray', alpha=dyn_alpha, label='Sample Differentials')
            ax[i, j].plot(x_test[1:, idx].flatten(), z_pred[1:, idx].flatten(), 'o', color='orange', alpha=0.5, label='DiffReg')
            ax[i, j].plot(x_test[1:, idx].flatten(), z_test[:, idx].flatten(), color='black', label='MC (bump & reval)')
            ax[i, j].set_facecolor(backgroundcolor[float(weights[idx])])

            if j == 0:
                ax[i, j].set_ylabel(fixRateLabel[float(moneynessPctList[i])])

            if i == 0:
                ax[i, j].set_title(float_to_time_str(maturityFloatList[j]))

            # Adjust size of price plot
            box = ax[i, j].get_position()
            ax[i, j].set_position([box.x0, box.y0, box.width, box.height * 0.8])

            ax[i, j].xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())

        # Legend
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, fancybox=True, shadow=True,
                   bbox_to_anchor=(0.5, 0.95))

    fig.suptitle('Basket Deltas')
    plt.savefig(get_plot_path('07_vasicek_ADD_DiffReg_delta_BasketEuPayerSwaption_mixed.png'), dpi=400)
    plt.show()


