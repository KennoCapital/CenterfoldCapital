import torch
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
from torch.utils.data import RandomSampler, DataLoader
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import BermudanPayerSwaption, EuropeanPayerSwaption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import RNG, LSMC, lsmcDefaultSim, lsmcPayoff, mcSimPaths, mcSim
from application.engine.regressor import PolynomialRegressor
from application.utils.path_config import get_plot_path
from application.experiments.vasicek.vasicek_hedge_tools import diff_reg_fit_predict
from application.utils.torch_utils import max0
from application.utils.prd_name_conventions import float_to_time_str

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024  # TODO decide on this
    N_test = 256
    n = 5000
    use_av = True

    hedge_times = 500

    r0_min = 0.02
    r0_max = 0.12

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Setup Differential Regressor, and Scalar
    deg = 9
    deg_lsmc = 9
    alpha = 1.0
    include_interactions = True
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True,
                                               include_interactions=include_interactions)

    # Model specification
    r0 = torch.linspace(r0_min, r0_max, N_test)
    a = torch.tensor(0.86)
    b = r0.median()
    sigma = torch.tensor(0.0148)
    measure = 'risk_neutral'

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

    strike = mdl.calc_swap_rate(r0.median(), t0_swap_fixings, delta)

    prd = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    idx_start = int(0.0 not in exerciseDates)
    t_swap_fixings = [sample.irs[0].fixingDates for i, sample in enumerate(prd.defline[idx_start:])]

    poly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=True, bias=True, include_interactions=include_interactions)
    lsmc = LSMC(reg=poly_reg)

    """ Helper functions for generating training data of pathwise payoffs and deltas """

    def calc_dswap_dr(r0_vec, t0):
        ones = torch.ones_like(r0_vec)
        res = []
        mask = exerciseDates > t0
        for t_fix in itertools.compress(t_swap_fixings, mask):
            def _swap_price(r0_vec):
                tau = t_fix - t0
                S = mdl.calc_swap(r0_vec, tau, delta, strike, notional)
                return S
            Jv = jvp(_swap_price, r0_vec, ones, create_graph=False)
            res.append(Jv)
        prices = torch.vstack([x[0] for x in res]).T
        derivs = torch.vstack([x[1] for x in res]).T
        return prices, derivs

    def calc_dBerSwpt_dr(r0_vec, t0):
        # TODO consider using different `n` and resample r0_vec
        mask = exerciseDates >= t0
        def _swpt(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = BermudanPayerSwaption(
                    strike=strike,
                    exerciseDates=exerciseDates[mask] - t0,
                    delta=delta,
                    swapLastFixingDate=swapLastFixingDate - t0,
                    notional=notional
                )
            cRng = RNG(seed=seed, use_av=use_av)
            cPoly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=True, bias=True, include_interactions=include_interactions)
            cLsmc = LSMC(reg=cPoly_reg)
            payoff = lsmcDefaultSim(prd=cPrd, mdl=cMdl, rng=cRng, N=len(r0_vec), n=len(r0_vec), lsmc=cLsmc)
            return torch.sum(payoff, dim=0)
        ones = torch.ones_like(r0_vec)
        prices, derivs = jvp(func=_swpt, inputs=r0_vec, v=ones, create_graph=False)
        return prices, derivs

    def calc_dEuSwpt_dr(r0_vec: torch.Tensor, t0: float):
        def _payoffs(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure=measure)
            cPrd = EuropeanPayerSwaption(
                    strike=strike,
                    exerciseDate=exerciseDates[-1] - t0,
                    delta=delta,
                    swapFirstFixingDate=t_swap_fixings[-1][0] - t0,
                    swapLastFixingDate=swapLastFixingDate - t0,
                    notional=notional
            )

            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec))
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res

    """ Delta Hedge Experiment """

    # Simulate paths
    # Sample (with replacement) `n` initial values of r0 from the r0_vec.
    # These `n` samples are used in the pre-simulation in order to estimate the early exercise boundary
    rngSampler = RandomSampler(r0_vec, replacement=True, num_samples=n, generator=torch.Generator().manual_seed(seed))
    dl = DataLoader(r0_vec, sampler=rngSampler, batch_size=n)
    r0_presim = [x for x in dl][0]
    mdl_presim = Vasicek(a, b, sigma, r0_presim)
    dTL = torch.linspace(0.0, float(exerciseDates[-1]), hedge_times + 1)
    preSimPaths = mcSimPaths(prd=prd, model=mdl_presim, rng=rng, N=n, dTL=dTL)
    paths = mcSimPaths(prd=prd, model=mdl, rng=rng, N=N_test, dTL=dTL)
    payoff = lsmcPayoff(prd=prd, preSimPaths=preSimPaths, paths=paths, lsmc=lsmc)

    exercise_idx = prd.exercise_idx
    r = mdl.x

    # Get price of claim (we use 500k simulations to get an accurate estimate)
    swpt = torch.empty_like(r[0, :])
    for i in tqdm(range(N_test), desc='Estimating initial value of Bermudan Swaption with MC'):
        mdl_pricer = Vasicek(a, b, sigma, r[0, i], measure='terminal')
        paths_pricer = mcSimPaths(prd=prd, model=mdl_pricer, rng=rng, N=50000)
        payoff_pricer = lsmcPayoff(prd=prd, preSimPaths=preSimPaths, paths=paths_pricer, lsmc=lsmc)
        swpt[i] = torch.mean(torch.sum(payoff_pricer, dim=0))

    # Initialize experiment
    swap = torch.vstack([mdl.calc_swap(r[0, :], t, delta, strike, notional) for t in t_swap_fixings]).T

    V = swpt
    h_a = diff_reg_fit_predict(u_vec=swap, r0_vec=r0_vec, t0=0.0,
                               calc_dPrd_dr=calc_dBerSwpt_dr, calc_dU_dr=calc_dswap_dr,
                               diff_reg=diff_reg, use_av=use_av)[1]
    h_b = V - torch.sum(h_a * swap, dim=1)

    swapTe = torch.full((len(exerciseDates), N_test), torch.nan)
    vTe = torch.full((len(exerciseDates), N_test), torch.nan)
    idx = 0

    # Loop over time
    mask = exerciseDates >= 0.0
    for k in tqdm(range(1, len(dTL)), desc='Running delta-hedge experiment'):
        dt = dTL[k] - dTL[k-1]
        t = dTL[k]

        # Update market variables
        swap = torch.vstack([mdl.calc_swap(r[k, :], t_fixing - t, delta, strike, notional)
                             if mask[i] else torch.full_like(r[k, :], torch.nan)
                             for i, t_fixing in enumerate(t_swap_fixings)]).T

        # Update portfolio
        V = torch.sum(h_a * swap[:, mask], dim=1) + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)

        # Save results on exercise dates
        if t in exerciseDates:
            swapTe[idx, exercise_idx == idx] = swap[exercise_idx == idx, idx]
            vTe[idx, exercise_idx == idx] = V[exercise_idx == idx]
            idx += 1

        if k < len(dTL) - 1:
            mask = exerciseDates > t
            calc_dPrd_dr = calc_dBerSwpt_dr if torch.sum(mask) > 1 else calc_dEuSwpt_dr
            r0_vec = choose_training_grid(r[k, :], N_train)
            h_a = diff_reg_fit_predict(u_vec=swap[:, mask], r0_vec=r0_vec, t0=t,
                                       calc_dPrd_dr=calc_dPrd_dr, calc_dU_dr=calc_dswap_dr,
                                       diff_reg=diff_reg, use_av=use_av)[1]
            h_b = V - torch.sum(h_a * swap[:, mask], dim=1)
        """
        if k in (50, 150, 250, 350, 450):
            if idx < 2:
                fig, ax = plt.subplots(3 - idx)
                for i in range(idx, 3):
                    ax[i-idx].plot(swap[:, i], h_a[:, i-idx], 'o')
                    ax[i-idx].set_ylabel(exerciseDates[i])
                fig.suptitle(f't={t}')
                plt.show()
            else:
                plt.plot(swap[:, idx], h_a.flatten(), 'o')
                plt.title(f't={t}')
                plt.show()
        """

    """ Plot results """
    MAE_value = torch.nanmean(torch.abs(vTe - max0(swapTe)))

    portPayoff = max0(swapTe)
    color = ['orange', 'blue', 'green']
    x = torch.linspace(float(torch.nansum(swapTe, dim=0).min()),
                       float(torch.nansum(swapTe, dim=0).max()),
                       1001)

    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.plot(x, max0(x), color='black', label='Payoff')
    for i, Te in enumerate(exerciseDates):
        ax.plot(swapTe[i, :], vTe[i, :], 'o', color=color[i], alpha=0.5, label=float_to_time_str(Te))

    ax.text(0.05, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax.transAxes)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
    ax.set_xlabel('Swap @ Exercise date')

    fig.suptitle('Replicating payoff of Bermudan Payer Swaption')
    fig.legend(title='Exercise date', loc='upper center', ncol=4, fancybox=True, shadow=True, bbox_to_anchor=(0.5, 0.925))

    plt.savefig(get_plot_path('07_Diff_reg_bermudan_delta_hedge.png'), dpi=400)
    plt.show()
