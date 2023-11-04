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
    test_set_filename = 'vasicek_BerPayerSwpt_test_set_086'
    filename_path = get_data_path(test_set_filename + '.pkl')

    seed = 1234
    N_train = 4096
    use_av = True

    test_bump_size_bp = torch.tensor(1.0)

    r0_min = torch.tensor(-0.02)
    r0_max = torch.tensor(0.15)

    N_test = torch.round((r0_max - r0_min) * test_bump_size_bp * 10000)

    r0_vec = torch.linspace(float(r0_min), float(r0_max), N_train)

    # Setup Differential Regressor, and Scalar
    deg_lsmc = 5
    deg = 7
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(
        deg=deg,
        alpha=alpha,
        use_SVD=True,
        bias=True,
        include_interactions=True
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
    exerciseDates = torch.tensor([1.0, 2.0, 5.0])
    delta = torch.tensor(0.25)
    swapLastFixingDate = torch.tensor(10.0)
    notional = torch.tensor(1e6)

    t0_swap_fixings = torch.linspace(
        float(exerciseDates[-1]),
        float(swapLastFixingDate),
        int((swapLastFixingDate - exerciseDates[-1]) / delta + 1)
    )

    strike = mdl.calc_swap_rate(r0, t0_swap_fixings, delta)

    prd = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    idx_start = int(0.0 not in exerciseDates)
    t_swap_fixings = [sample.irs[0].fixingDates for i, sample in enumerate(prd.defline[idx_start:])]

    poly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=True)
    lsmc = LSMC(reg=poly_reg)

    """ Helper functions for generating training data of pathwise payoffs and deltas """

    def calc_dswap_dr(r0_vec, t0):
        ones = torch.ones_like(r0_vec)
        res = []
        for i in range(len(exerciseDates)):
            def _swap_price(r0_vec):
                tau = t_swap_fixings[i] - t0
                S = mdl.calc_swap(r0_vec, tau, delta, strike, notional)
                return S
            Jv = jvp(_swap_price, r0_vec, ones, create_graph=False)
            res.append(Jv)
        prices = torch.vstack([x[0] for x in res]).T
        derivs = torch.vstack([x[1] for x in res]).T
        return prices, derivs

    def calc_dswpt_dr(r0_vec, t0):
        def _swpt(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = BermudanPayerSwaption(
                    strike=strike,
                    exerciseDates=exerciseDates - t0,
                    delta=delta,
                    swapLastFixingDate=swapLastFixingDate - t0,
                    notional=notional
                )
            cRng = RNG(seed=seed, use_av=use_av)
            cPoly_reg = PolynomialRegressor(deg=deg_lsmc)
            cLsmc = LSMC(reg=cPoly_reg)
            payoff = lsmcDefaultSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec), n=len(r0_vec), lsmc=cLsmc)
            return torch.sum(payoff, dim=0)
        ones = torch.ones_like(r0_vec)
        prices, derivs = jvp(func=_swpt, inputs=r0_vec, v=ones, create_graph=False)
        return prices, derivs


    """ Calculate `true` swaption price using Monte Carlo for comparison """
    if test_set_filename is not None and os.path.isfile(filename_path):
        with open(filename_path, "rb") as input_file:
            x_test, y_test, z_test = pickle.load(input_file)
    else:
        r0_test_vec = torch.linspace(float(r0_min), float(r0_max), int(N_test) + 1)
        x_test = torch.hstack(
            [mdl.calc_swap(r0_test_vec, t_swap_fixings[i], delta, strike, notional).reshape(-1, 1)
             for i in range(len(exerciseDates))]
        )
        y_test = torch.full_like(r0_test_vec, torch.nan)
        for j in tqdm(range(len(r0_test_vec)), desc='Calculating pricing of BerPayerSwpt using MC'):
            tmp_mdl = Vasicek(a, b, sigma, r0_test_vec[j], use_ATS=True, use_euler=False, measure='terminal')
            tmp_rng = RNG(seed=seed, use_av=True)
            payoff = lsmcDefaultSim(prd=prd, mdl=tmp_mdl, rng=tmp_rng, N=50000, n=5000, lsmc=lsmc, reg=poly_reg)
            y_test[j] = torch.mean(torch.sum(payoff, dim=0))
        y_test = y_test.reshape(-1, 1)
        z_test = y_test.diff(dim=0) / x_test.diff(dim=0)

        # Export results
        if test_set_filename is not None:
            with open(filename_path, 'wb') as output_file:
                pickle.dump((x_test, y_test, z_test), output_file, protocol=pickle.HIGHEST_PROTOCOL)

    """ Estimate Price and Delta using Differential Regression """

    x_train, y_train, z_train = training_data(
        r0_vec=r0_vec, t0=0.0, calc_dU_dr=calc_dswap_dr, calc_dPrd_dr=calc_dswpt_dr, use_av=use_av
    )

    x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)

    diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)

    x_test_scaled, _, _ = scalar.transform(x_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(x_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    RMSE_price = torch.sqrt(torch.mean((y_pred - y_test) ** 2))
    MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_test))

    fig, ax = plt.subplots(nrows=2, ncols=len(exerciseDates), sharex='col')

    for i in range(len(exerciseDates)):
        ax[0, i].plot(x_train[:, i].flatten(), y_train.flatten(), 'o', color='gray', alpha=0.25)
        ax[0, i].plot(x_test[:, i].flatten(), y_test, color='black')
        ax[0, i].plot(x_test[:, i].flatten(), y_pred, 'o', color='orange', alpha=0.5)

        ax[1, i].plot(x_train[:, i].flatten(), z_train[:, i].flatten(), 'o', color='gray', alpha=0.25)
        ax[1, i].plot(x_test[1:, i].flatten(), z_test[:, i].flatten(), color='black')
        ax[1, i].plot(x_test[1:, i].flatten(), z_pred[1:, i].flatten(), 'o', color='orange', alpha=0.5)

    plt.show()




