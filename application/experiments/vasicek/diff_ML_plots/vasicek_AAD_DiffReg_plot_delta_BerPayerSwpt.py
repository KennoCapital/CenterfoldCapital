import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp, jacobian
from application.engine.vasicek import Vasicek
from application.engine.products import BermudanPayerSwaption
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import RNG, LSMC, lsmcDefaultSim
from application.engine.regressor import PolynomialRegressor
from application.utils.path_config import get_plot_path
from tqdm import tqdm

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_presim = 16384
    N_train = 65536
    N_test = 256
    use_av = True

    r0_min = -0.02
    r0_max = 0.15

    r0_vec = torch.linspace(r0_min, r0_max, 101)

    # Setup Differential Regressor, and Scalar
    deg_lsmc = 5
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

    payoff = lsmcDefaultSim(
        prd=prd, mdl=mdl, rng=rng, N=N_train, n=N_presim, lsmc=lsmc, reg=poly_reg
    )

    price_bermudan_payer_swpt = torch.mean(torch.sum(payoff, dim=0))

    """ Helper functions for generating training data of pathwise payoffs and deltas """

    def calc_dswpt_dparam(r0_vec):
        def _swpt(r0_vec):
            mdl_ = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            rng_ = RNG(seed=seed, use_av=use_av)
            poly_reg_ = PolynomialRegressor(deg=deg_lsmc)
            lsmc_ = LSMC(reg=poly_reg_)
            payoff_ = lsmcDefaultSim(prd=prd, mdl=mdl_, rng=rng_, N=50000, n=5000, lsmc=lsmc_)
            return torch.mean(torch.sum(payoff_, dim=0))
        ones = torch.ones_like(r0_vec)
        J = jvp(func=_swpt, inputs=r0_vec, v=ones, create_graph=False)
        return J

    # res = calc_dswpt_dparam(r0_vec)

    """ Calculate `true` swaption price using Monte Carlo for comparison """
    r0_test_vec = torch.linspace(r0_min, r0_max, N_test)
    X_test = mdl.calc_swap(r0_test_vec, t_swap_fixings, delta, strike, notional).reshape(-1, 1)
    y_mdl = torch.full_like(r0_test_vec, torch.nan)
    for j in tqdm(range(len(r0_test_vec))):
        tmp_mdl = Vasicek(a, b, sigma, r0_test_vec[j], use_ATS=True, use_euler=False, measure='terminal')
        tmp_rng = RNG(seed=seed, use_av=True)
        payoff = lsmcDefaultSim(prd=prd, mdl=tmp_mdl, rng=tmp_rng, N=50000, n=5000, lsmc=lsmc, reg=poly_reg)
        y_mdl[j] = torch.mean(torch.sum(payoff, dim=0))
    y_mdl = y_mdl.reshape(-1, 1)
    z_mdl = y_mdl.diff(dim=0) / X_test.diff(dim=0)
