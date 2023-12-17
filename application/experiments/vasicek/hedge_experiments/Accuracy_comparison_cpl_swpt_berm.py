import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import CapletAsPutOnZCB, EuropeanPayerSwaption, BermudanPayerSwaption
from application.engine.mcBase import mcSimPaths, mcSim, RNG, LSMC, lsmcDefaultSim, lsmcPayoff
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.regressor import PolynomialRegressor
from application.utils.path_config import get_plot_path
from torch.utils.data import RandomSampler, DataLoader
from application.utils.torch_utils import max0
from application.experiments.vasicek.vasicek_hedge_tools import diff_reg_fit_predict, calc_delta_diff_nn, calc_delta_diff_nn_bermudan

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


if __name__ == '__main__':

    ## Structure:
        # hedge caplet
        # hedge swaption
        # hedge bermudan
        # plot results that is appended throughout the hedging

    # hedge setting
    resList = []
    labelList = []

    diffML = ['diffreg', 'diffNN']

    Training_samples = [1024] #[8, 16, 32, 64, 128, 256, 512, 1024, 4096] #, 8196] #, 8196 * 2]

    seed = 1234
    N_test = 256
    use_av = True

    r0_min = -0.02
    r0_max = 0.15

    # Model specification
    r0 = torch.tensor(0.08)
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    strike = torch.tensor(0.0871)

    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    """CAPLET HEDGING"""

    # Product specification
    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )


    """ Helper functions for generating training data of pathwise payoffs and deltas """
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


    def get_diffML_delta(ML : str, tau : float = 0.0):
        if ML == 'diffreg':
            return diff_reg_fit_predict(u_vec=U, r0_vec=r0_train_vec, t0=tau,
                                        calc_dPrd_dr=calc_dcpl_dr, calc_dU_dr=calc_dzcb_dr,
                                        diff_reg=diff_reg, use_av=use_av)[1].flatten()
        elif ML == 'diffNN':
            return calc_delta_diff_nn(u_vec=U, r0_vec=r0_train_vec, t0=tau,
                                         calc_dPrd_dr=calc_dcpl_dr, calc_dU_dr=calc_dzcb_dr,
                                         nn_Params=nn_params, use_av=use_av)
        else:
            raise ValueError("Wrongly specified ML method")




    """ Delta Hedge Experiment """

    steps = 250 // 2
    dTL = torch.linspace(0.0, float(exerciseDate), steps + 1)

    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    for ML in diffML:
        for N_train in tqdm(Training_samples, desc=f'{ML} for Caplet'):
            r0_train_vec = torch.linspace(r0_min, r0_max, N_train)

            if ML == 'diffreg':
                # Setup Differential Regressor
                diff_reg = DifferentialPolynomialRegressor(deg=9, alpha=1.0, use_SVD=True, bias=True,
                                                           include_interactions=True)
            else:
                # Differential Neural Network Settings
                seed_weights = 1234
                epochs = 250
                batches_per_epoch = 16
                hidden_units = 20
                hidden_layers = 4
                lam = 1.0
                min_batch_size = int(N_train * 5 / 8)

                nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                             'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_size,
                             'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

            # Get price of claim (we use 500k simulations to get an accurate estimate)
            if r0.dim() != 0:
                V = torch.empty_like(r[0, :])
                for n in range(N_test):
                    mdl.r0 = r[0, n]
                    V[n] = torch.mean(mcSim(prd, mdl, rng, 50000))
            else:
                price = torch.mean(mcSim(prd, mdl, rng, 500000))
                V = price * torch.ones(N_test)

            # Initialize experiment
            U = mdl.calc_zcb(mdl.r0 * torch.ones_like(V), exerciseDate + delta)[0]

            h_a = get_diffML_delta(ML, tau=0.0)

            h_b = V - h_a * U

            # Loop over time
            for k in range(1, len(dTL)):
                dt = dTL[k] - dTL[k - 1]
                t = dTL[k]

                # Update market variables
                U = mdl.calc_zcb(r[k, :], exerciseDate + delta - t)[0]

                # Update portfolio
                V = h_a * U + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
                if k < len(dTL) - 1:
                    # r0_vec = choose_training_grid(r[k, :], N)
                    h_a = get_diffML_delta(ML, tau=t)
                    h_b = V - h_a * U

            resList.append(torch.std(V - max0(U)))
            labelList.append(("Caplet", ML, N_train))


    """EUROPEAN SWAPTION HEDGING"""
    # Product specification
    exerciseDate = torch.tensor(5.0)
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


    def get_diffML_delta(ML : str, tau : float = 0.0):
        if ML == 'diffreg':
            return diff_reg_fit_predict(u_vec=U, r0_vec=r0_train_vec, t0=tau,
                                        calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                        diff_reg=diff_reg, use_av=use_av)[1].flatten()
        elif ML == 'diffNN':
            return calc_delta_diff_nn(u_vec=U, r0_vec=r0_train_vec, t0=tau,
                                         calc_dPrd_dr=calc_dswpt_dr, calc_dU_dr=calc_dswap_dr,
                                         nn_Params=nn_params, use_av=use_av)
        else:
            raise ValueError("Wrongly specified ML method")


    """ Delta Hedge Experiment """
    steps = 250 // 2
    dTL = torch.linspace(0.0, float(exerciseDate), steps + 1)

    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    for ML in diffML:
        for N_train in tqdm(Training_samples, desc=f'{ML} for EurSwpt'):
            r0_train_vec = torch.linspace(r0_min, r0_max, N_train)

            if ML == 'diffreg':
                # Setup Differential Regressor
                diff_reg = DifferentialPolynomialRegressor(deg=9, alpha=1.0, use_SVD=True, bias=True,
                                                           include_interactions=True)
            else:
                # Differential Neural Network Settings
                seed_weights = 1234
                epochs = 250
                batches_per_epoch = 16
                hidden_units = 20
                hidden_layers = 4
                lam = 1.0
                min_batch_size = int(N_train * 5 / 8)

                nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                             'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_size,
                             'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

            # Get price of claim (we use 500k simulations to get an accurate estimate)
            if r0.dim() != 0:
                V = torch.empty_like(r[0, :])
                for n in range(N_test):
                    mdl.r0 = r[0, n]
                    V[n] = torch.mean(mcSim(prd, mdl, rng, 50000))
            else:
                price = torch.mean(mcSim(prd, mdl, rng, 500000))
                V = price * torch.ones(N_test)

            # Initialize experiment
            U = mdl.calc_swap(mdl.r0 * torch.ones_like(V), t_swap_fixings, delta, strike, notional)

            h_a = get_diffML_delta(ML, tau=0.0)

            h_b = V - h_a * U

            # Loop over time
            for k in range(1, len(dTL)):
                dt = dTL[k] - dTL[k - 1]
                t = dTL[k]

                # Update market variables
                U = mdl.calc_swap(r[k, :], t_swap_fixings - t, delta, strike, notional)

                # Update portfolio
                V = h_a * U + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
                if k < len(dTL) - 1:
                    # r0_vec = choose_training_grid(r[k, :], N)
                    h_a = get_diffML_delta(ML, tau=t)
                    h_b = V - h_a * U

            resList.append(torch.std(V - max0(U)))
            labelList.append(("EurSwpt", ML, N_train))


    """Bermudan Swaption Hedging"""

    deg_lsmc = 9
    n = 5000

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

    prd = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    idx_start = int(0.0 not in exerciseDates)
    t_swap_fixings = [sample.irs[0].fixingDates for i, sample in enumerate(prd.defline[idx_start:])]

    poly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=True, bias=True, include_interactions=True)
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
            cPoly_reg = PolynomialRegressor(deg=deg_lsmc, use_SVD=True, bias=True, include_interactions=True)
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


    def get_diffML_delta(ML : str, U, calc_dPrd_dr, tau : float = 0.0):
        if ML == 'diffreg':
            return diff_reg_fit_predict(u_vec=U, r0_vec=r0_train_vec, t0=tau,
                                        calc_dPrd_dr=calc_dPrd_dr, calc_dU_dr=calc_dswap_dr,
                                        diff_reg=diff_reg, use_av=use_av)[1]
        elif ML == 'diffNN':
            return calc_delta_diff_nn_bermudan(u_vec=U, r0_vec=r0_train_vec, t0=tau,
                                         calc_dPrd_dr=calc_dPrd_dr, calc_dU_dr=calc_dswap_dr,
                                         nn_Params=nn_params, use_av=use_av)
        else:
            raise ValueError("Wrongly specified ML method")


    """ Delta Hedge Experiment """
    steps = 250 // 2
    dTL = torch.linspace(0.0, float(exerciseDates[-1]), steps + 1)

    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x


    for ML in diffML:
        for N_train in tqdm(Training_samples, desc=f'{ML} for BerSwpt'):
            r0_train_vec = torch.linspace(r0_min, r0_max, N_train)

            # Simulate paths
            # Sample (with replacement) `n` initial values of r0 from the r0_vec.
            # These `n` samples are used in the pre-simulation in order to estimate the early exercise boundary
            rngSampler = RandomSampler(r0_train_vec, replacement=True, num_samples=n,
                                       generator=torch.Generator().manual_seed(seed))
            dl = DataLoader(r0_train_vec, sampler=rngSampler, batch_size=n)
            r0_presim = [x for x in dl][0]
            mdl_presim = Vasicek(a, b, sigma, r0_presim)
            dTL = torch.linspace(0.0, float(exerciseDates[-1]), steps + 1)
            preSimPaths = mcSimPaths(prd=prd, model=mdl_presim, rng=rng, N=n, dTL=dTL)
            paths = mcSimPaths(prd=prd, model=mdl, rng=rng, N=N_test, dTL=dTL)
            payoff = lsmcPayoff(prd=prd, preSimPaths=preSimPaths, paths=paths, lsmc=lsmc)

            exercise_idx = prd.exercise_idx

            if ML == 'diffreg':
                # Setup Differential Regressor
                diff_reg = DifferentialPolynomialRegressor(deg=9, alpha=1.0, use_SVD=True, bias=True,
                                                           include_interactions=True)
            else:
                # Differential Neural Network Settings
                seed_weights = 1234
                epochs = 250
                batches_per_epoch = 16
                hidden_units = 20
                hidden_layers = 4
                lam = 1.0
                min_batch_size = int(N_train * 5 / 8)

                nn_params = {'N_train': N_train, 'seed_weights': seed_weights, 'epochs': epochs,
                             'batches_per_epoch': batches_per_epoch, 'min_batch_size': min_batch_size,
                             'lam': lam, 'hidden_units': hidden_units, 'hidden_layers': hidden_layers}

            if r0.dim() != 0:
                V = torch.empty_like(r[0, :])
                for n in range(N_test):
                    mdl_pricer = Vasicek(a, b, sigma, r[0, n], measure='terminal')
                    paths_pricer = mcSimPaths(prd=prd, model=mdl_pricer, rng=rng, N=50000)
                    payoff_pricer = lsmcPayoff(prd=prd, preSimPaths=preSimPaths, paths=paths_pricer, lsmc=lsmc)
                    V[n] = torch.mean(torch.sum(payoff_pricer, dim=0))
            else:
                paths_pricer = mcSimPaths(prd=prd, model=mdl, rng=rng, N=500000)
                payoff_pricer = lsmcPayoff(prd=prd, preSimPaths=preSimPaths, paths=paths_pricer, lsmc=lsmc)
                price = torch.mean(torch.sum(payoff_pricer, dim=0))
                V = price * torch.ones(N_test)

            swap = torch.vstack([mdl.calc_swap(r[0, :], t, delta, strike, notional) for t in t_swap_fixings]).T

            h_a = get_diffML_delta(ML=ML, U=swap, calc_dPrd_dr=calc_dBerSwpt_dr, tau=0.0)

            h_b = V - torch.sum(h_a * swap, dim=1)

            swapTe = torch.full((len(exerciseDates), N_test), torch.nan)
            vTe = torch.full((len(exerciseDates), N_test), torch.nan)
            idx = 0

            mask = exerciseDates >= 0.0
            for k in range(1, len(dTL)):
                dt = dTL[k] - dTL[k - 1]
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
                    h_a = get_diffML_delta(ML=ML, U=swap[:, mask], calc_dPrd_dr=calc_dPrd_dr, tau=t)

                    h_b = V - torch.sum(h_a * swap[:, mask], dim=1)

            difference = vTe - max0(swapTe)
            mask = ~torch.isnan(difference)
            mean = torch.mean(difference[mask])
            std_value = torch.sqrt(torch.mean((difference[mask] - mean) ** 2))
            resList.append(std_value.item())
            labelList.append(("BerSwpt", ML, N_train))




    """SUMMARIZE HEDGE EXPERIMENTS"""

    import pandas as pd
    resList_np = [tensor.item() for tensor in resList]

    # Create a DataFrame for easier handling
    df = pd.Series(resList_np, index=pd.MultiIndex.from_tuples(labelList, names=["PRD", "DIFFML", "TRAINING_SAMPLES"])).reset_index()
    df.rename(columns={0: 'Value'}, inplace=True)

    """
    # Plotting
    plt.figure()
    for prd in df['PRD'].unique():
        for diffml in df['DIFFML'].unique():
            subset = df[(df['PRD'] == prd) & (df['DIFFML'] == diffml)]
            plt.plot(np.log(subset['TRAINING_SAMPLES']), np.log(subset['Value']),
                     '-o',
                     label=prd,
                     color='orange' if diffml == 'diffreg' else 'black')

            plt.annotate(prd,
                         xy=(np.log(subset['TRAINING_SAMPLES'].iloc[-1]), np.log(subset['Value'].iloc[-1])),
                         xytext=(-30, 5), # offset x and y a little to keep labels within the frame
                         textcoords='offset points',
                         color='orange' if diffml == 'diffreg' else 'black'
                         )

    legend_elements = [plt.Line2D([0], [0], color='orange', lw=2, label='Diff. Reg'),
                       plt.Line2D([0], [0], color='black', lw=2, label='Diff. NN')]
    plt.legend(handles=legend_elements, draggable=True, frameon=False)
    plt.title(f'Hedge Times={steps}')
    plt.xlabel('Training Samples')
    plt.ylabel('Std. of Hedge Error')
    plt.xticks(ticks=np.log(Training_samples), labels=Training_samples)
    plt.show()
    """




