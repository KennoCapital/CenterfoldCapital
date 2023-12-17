import torch
import time
import numpy as np
from torch.autograd.functional import jvp
from tqdm import tqdm
from application.engine.vasicek import Vasicek
from application.engine.products import CapletAsPutOnZCB, EuropeanPayerSwaption, BermudanPayerSwaption
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.regressor import PolynomialRegressor
from application.engine.differential_NN import Neural_Approximator
from application.engine.mcBase import mcSimPaths, mcSim, RNG, LSMC, lsmcDefaultSim
from application.utils.path_config import get_plot_path
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import training_data

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':



    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True

    hedge_points = 250

    r0_min = 0.00
    r0_max = 0.16

    r0_train_vec = torch.linspace(r0_min, r0_max, N_train)

    # Model specification
    r0 = torch.linspace(r0_min, r0_max, N_test)
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = mdl.calc_swap_rate(r0.median(), exerciseDate, delta)


    """CAPLET"""

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
            cTL = dTL[dTL > exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec), cTL)
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res


    X_train, y_train, z_train = training_data(r0_vec=r0_train_vec, t0=0.0, calc_dU_dr=calc_dzcb_dr, calc_dPrd_dr=calc_dcpl_dr)


    """Regression"""

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    duration = []
    for i in range(100):
        start_time = time.time()

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)
        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)
        X_test_scaled, _, _ = scalar.transform(X_train, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)
        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        end_time = time.time()

        duration.append(end_time - start_time)



    """Neural Network"""
    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = int(N_train * 5 / 8)
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4


    duration = []
    for i in tqdm(range(20)):
        start_time = time.time()

        diff_nn = Neural_Approximator(X_train, y_train, z_train)
        diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                        hidden_layers=hidden_layers)
        diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
        y_pred, z_pred = diff_nn.predict_values_and_derivs(X_train)

        end_time = time.time()

        duration.append(end_time - start_time)




    """EUROPEAN SWAPTION"""

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

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
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


    X_train, y_train, z_train = training_data(r0_vec=r0_train_vec, t0=0.0, calc_dU_dr=calc_dswap_dr,
                                              calc_dPrd_dr=calc_dswpt_dr)

    """Regression"""
    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    duration = []
    for i in range(100):
        start_time = time.time()

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)
        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)
        X_test_scaled, _, _ = scalar.transform(X_train, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)
        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        end_time = time.time()

        duration.append(end_time - start_time)

    """Neural Network"""
    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = int(N_train * 5 / 8)
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4

    duration = []
    for i in tqdm(range(20)):
        start_time = time.time()

        diff_nn = Neural_Approximator(X_train, y_train, z_train)
        diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                        hidden_layers=hidden_layers)
        diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
        y_pred, z_pred = diff_nn.predict_values_and_derivs(X_train)

        end_time = time.time()

        duration.append(end_time - start_time)




    """BERMUDAN SWAPTION"""

    deg_lsmc = 5
    deg = 5
    alpha = 1.0
    use_SVD = True
    bias = True
    include_interactions = True

    exerciseDates = torch.tensor([1.0, 2.0, 5.0])
    delta = torch.tensor(0.25)
    swapLastFixingDate = torch.tensor(10.0)
    notional = torch.tensor(1e6)

    t0_swap_fixings = torch.linspace(
        float(exerciseDates[-1]),
        float(swapLastFixingDate),
        int((swapLastFixingDate - exerciseDates[-1]) / delta + 1)
    )

    prd = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    diff_reg = DifferentialPolynomialRegressor(
        deg=deg,
        alpha=alpha,
        use_SVD=use_SVD,
        bias=bias,
        include_interactions=include_interactions
    )
    scalar = DifferentialStandardScaler()

    idx_start = int(0.0 not in exerciseDates)
    t_swap_fixings = [sample.irs[0].fixingDates for i, sample in enumerate(prd.defline[idx_start:])]

    poly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=use_SVD, bias=bias, include_interactions=include_interactions)
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
            cPoly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=use_SVD, bias=bias,
                                            include_interactions=include_interactions)
            cLsmc = LSMC(reg=cPoly_reg)
            payoff = lsmcDefaultSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec), n=len(r0_vec), lsmc=cLsmc)
            return torch.sum(payoff, dim=0)

        ones = torch.ones_like(r0_vec)
        prices, derivs = jvp(func=_swpt, inputs=r0_vec, v=ones, create_graph=False)
        return prices, derivs


    x_train, y_train, z_train = training_data(
        r0_vec=r0_train_vec, t0=0.0, calc_dU_dr=calc_dswap_dr, calc_dPrd_dr=calc_dswpt_dr, use_av=use_av
    )

    """Regression"""
    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    duration = []
    for i in range(100):
        start_time = time.time()

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)
        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)
        X_test_scaled, _, _ = scalar.transform(X_train, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)
        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        end_time = time.time()

        duration.append(end_time - start_time)




    """Neural Network"""
    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = int(N_train * 5 / 8)
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4

    duration = []
    for i in tqdm(range(20)):
        start_time = time.time()

        diff_nn = Neural_Approximator(x_train, y_train, z_train)
        diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                        hidden_layers=hidden_layers)
        diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
        y_pred, z_pred = diff_nn.predict_values_and_derivs(X_train)

        end_time = time.time()

        duration.append(end_time - start_time)











