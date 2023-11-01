import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import Caplet
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.path_config import get_plot_path
from application.utils.torch_utils import max0

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


if __name__ == '__main__':

    seed = 1234
    N_train = 1024*8
    N_test = 1024
    use_av = True

    hedge_points = 100#25

    # Setup Differential Regressor, and Scalar
    deg = 15
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'

    def mean(t, r0, a=a, b=b):
        return r0 * torch.exp(-a*t) + b*(1-torch.exp(-a*t))

    def var(t, sigma=sigma, a= a):
        return sigma**2 / (2*a) * (1. - torch.exp(-2 * a * t))


    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(.25)
    delta = torch.tensor(.25)
    notional = torch.tensor(1.0)

    strike = mdl.calc_swap_rate(r0, exerciseDate, delta)

    r0_min = r0 - 5*var(exerciseDate+delta).sqrt()
    r0_max = r0 + 5*var(exerciseDate+delta).sqrt()

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    prd = Caplet(
        strike=strike,
        start=exerciseDate,
        delta=delta
    )

    # Simulate paths
    dTL = torch.linspace(0.0, float(exerciseDate), hedge_points + 1)
    mcSimPaths(prd, mdl, rng, N_test, dTL)
    r = mdl.x

    # Find index of the exercise date
    last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])

    """ Helper functions for calculating pathwise payoffs and deltas, and generating training data """
    def calc_dfwd_dr(r0_vec, t0):
        """
        :param  r0_vec:    Current Short rate r0
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """

        def _cpl(r0_vec):
            fwd = mdl.calc_fwd(r0_vec, exerciseDate - t0, delta)[0]
            return fwd

        ones = torch.ones_like(r0_vec)
        res = jvp(_cpl, r0_vec, ones, create_graph=False)
        return res

    def calc_dPdr(r0_vec, t0):

        def _P(r0_vec):
            P = mdl.calc_zcb(r0_vec, exerciseDate + delta - t0)
            return P
        ones = torch.ones_like(r0_vec)
        res = jvp(_P, r0_vec, ones, create_graph=False)
        return res


    def calc_dcpl_dr(r0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """

        def _payoffs(r0_vec):
            cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='terminal')
            cPrd = Caplet(
                strike=strike,
                start=exerciseDate - t0,
                delta=delta
            )
            payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec))
            return payoffs

        ones = torch.ones_like(r0_vec)
        res = jvp(_payoffs, r0_vec, ones, create_graph=False)
        return res


    def training_data(r0_vec: torch.Tensor, t0: float = 0.0, use_av: bool = True):

        """
        if t0 > 0.0:
            m = Normal(loc=mean(t0, r0), scale=var(t0).sqrt())
            r0_vec = m.sample_n(N_train)
        """

        if use_av:
            # X_train[i] = X_train[i + N_train],  for all i, when using AV
            r0_vec = torch.concat([r0_vec, r0_vec])

        P, dPdr = calc_dPdr(r0_vec, t0)
        fwd, dSdr = calc_dfwd_dr(r0_vec, t0)
        y, dydr = calc_dcpl_dr(r0_vec, t0)
        #PtT, dPTdr = calc_dPdr(r0_vec, t0, exerciseDate)
        #PtTD, dPTDdr =  calc_dPdr(r0_vec, t0)

        X_train = r0_vec.reshape(-1, 1)
        y_train = y.reshape(-1, 1)
        #y_train = (y * mdl.calc_zcb(r0_vec, exerciseDate + delta )).reshape(-1,1)

        #z_train = (dydr / dSdr ).reshape(-1, 1)

        z_train = ( dydr).reshape(-1, 1)

        #z_train =  (dydr / dPdr ).reshape(-1, 1)

        """
        #denominator = PtTD * dSdr
        #term1 = dydr - dPTdr * max0(fwd-strike) * PtT * PtTD
        #term2 = -PtT * max0(fwd-strike) * (dPTdr * PtTD + PtT*dPTDdr)
        # z_train = ((term1 + term2)/denominator * PtTD).reshape(-1, 1)

        #fwdT = mdl.calc_fwd(r[-1,], exerciseDate, delta)[0]
        nom = dydr - dPTDdr * max0(fwd-strike)
        denom = dSdr #* torch.exp(-a*(exerciseDate-t0))

        #z_train = (nom / denom ).reshape(-1, 1)
        """

        if use_av:
            idx_half = N_train
            X_train = X_train[:idx_half]
            y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
            z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])

        return X_train, y_train, z_train

    def calc_delta(fwd_vec: torch.Tensor, r0_vec: torch.Tensor, t0: float, use_av: bool) -> torch.Tensor:
        """
        param fwd_vec:      1D vector of market variables (Fwd prices)
        param r0_vec:       1D vector of short rates to generate training data from (swap prices)
        param t0:           Current market time, effect time to expiry and fixings in the training
        param use_av:       Use antithetic variates to reduce variance of both y- and z-labels

        returns:            1D vector of predicted deltas for `swap_vec`
        """
        X_test = fwd_vec.reshape(-1, 1)

        X_train, y_train, z_train = training_data(r0_vec=r0_vec, t0=t0, use_av=use_av)

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)
        #diff_reg.fit(X_train, y_train, z_train)

        X_test_scaled, _, _ = scalar.transform(X_test, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)
        #y_pred, z_pred = diff_reg.predict(X_test, predict_derivs=True)

        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        return z_pred.flatten()

    def calc_delta_bump_and_reval(r0_vec: torch.Tensor, t0: float, delta: torch.tensor, strike: torch.tensor, bump: float = 0.0001) -> torch.Tensor:
        """

        returns:            1D vector of predicted deltas
        """

        r_bump = r0_vec + bump
        fwd = mdl.calc_fwd(r0_vec, t0, delta)[0]
        fwd_bump = mdl.calc_fwd(r_bump, t0, delta)[0]
        cpl = mdl.calc_cpl(r0_vec, t0, delta, strike)[0]
        cpl_bump = mdl.calc_cpl(r_bump, t0, delta, strike)[0]

        z = (cpl_bump - cpl) / (fwd_bump - fwd)

        return z

    """ Delta Hedge Experiment """

    # Get price of claim (no need to simulate as we have an analytical expression)
    # cpl = mdl.calc_cpl(r0, exerciseDate, delta, strike)[0]
    cpl = torch.mean(mcSim(prd, mdl, rng, 500000))

    # Initialize experiment
    fwd = mdl.calc_fwd(r[0, :], exerciseDate, delta)[0]
    shortrate = r[0, :]
    zcb = mdl.calc_zcb(r[0,:], exerciseDate + delta)[0]

    h_a_lst = []
    h_b_lst = []
    V_lst = []
    cpl_lst = []

    cpl_lst.append(cpl)

    V = cpl * torch.ones_like(r[0, :])
    #h_a = calc_delta_bump_and_reval(r[0], exerciseDate - 0.0, delta, strike)
    #h_a = calc_delta(fwd_vec=fwd, r0_vec=r0_vec, t0=0.0, use_av=use_av)
    h_a = calc_delta(fwd_vec=shortrate, r0_vec=r0_vec, t0=0.0, use_av=use_av)

    B = torch.ones_like(r[0, :])
    #B = mdl.calc_zcb(r[0, :], torch.tensor(0.0100))[0]
    h_b = (V - h_a * shortrate) / B

    V_lst.append(V.mean())

    # Loop over time
    for k in range(1, last_idx+1):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        # Update market variables
        fwd = mdl.calc_fwd(r[k, :], exerciseDate - t, delta)[0]
        shortrate = r[k, :]

        cpl = mdl.calc_cpl(r[k, :], exerciseDate - t, delta, strike)[0]
        cpl_lst.append(cpl.mean())

        # Update portfolio
        #V = h_a * fwd + h_b * torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        B *= torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)

        V = h_a * shortrate + h_b * B
        V_lst.append(V.mean())

        zcb = mdl.calc_zcb(r[k, :], exerciseDate + delta - t)[0]

        if k < last_idx:
            #h_a = calc_delta_bump_and_reval(r[k, :], exerciseDate - t, delta, strike)
            rt_vec = torch.linspace(r[k, :].min(), r[k, :].max(), N_train//2)

            h_a = (calc_delta(fwd_vec=shortrate, r0_vec=rt_vec, t0=t, use_av=use_av))
            #h_a = max0(calc_delta(fwd_vec=zcb, r0_vec=rt_vec, t0=t, use_av=use_av))

            h_b = (V - h_a * shortrate) / B

            h_a_lst.append(h_a)
            h_b_lst.append(h_b)

    # fwdT = torch.linspace(float(fwd.min()), float(fwd.max()), 1001)
    fwdT = mdl.calc_fwd(r[last_idx, ].sort().values, torch.tensor(0.0), delta)[0]
    df = mdl.calc_zcb(r[last_idx, ].sort().values, delta)[0]
    payoff_func = delta * max0(fwdT - strike) * df

    MAE_value = torch.mean(torch.abs(V - payoff_func))

    # variance in holdings
    h_a_lst = torch.stack(h_a_lst)
    h_b_lst = torch.stack(h_b_lst)


    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots(1)
    ax.plot(fwdT, payoff_func, color='black', label='Payoff function')
    ax.plot(fwd, V, 'o', color='orange', label='Value of Hedge Portfolio', alpha=0.2)
    ax.set_xlabel('Fwd(T)')
    ax.text(0.05, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax.transAxes)

    # Adjust size of plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    fig.suptitle( 'CplSR\n' + f'\nHedgeFreq={dTL[1]:.4g}, alpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str + f'\n exercise={exerciseDate}, delta= {delta}')

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=2, fancybox=True, shadow=True),
               #bbox_to_anchor=(0.5, 0.90))

    plt.savefig(get_plot_path('vasicek_AAD_DiffReg_Caplet_delta_hedge.png'), dpi=400)
    plt.show()


    plt.figure()
    plt.plot(torch.stack(V_lst)- torch.stack(cpl_lst))
    plt.show()

    fwd_lst = []
    dFdr_lst =[]

    for k in range(1, last_idx + 1):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        rt_vec = torch.linspace(r[k, :].min(), r[k, :].max(), N_train//2)

        fwd, dFdr = calc_dfwd_dr(rt_vec, t)
        P, dPdr = calc_dPdr(rt_vec, t)
        y, dydr = calc_dcpl_dr(rt_vec, t)


        fwd_lst.append(fwd)
        dFdr_lst.append(dFdr)

        if k % 5 == 0:
            plt.figure()
            plt.scatter(rt_vec, dFdr)
            plt.xlabel('r')
            plt.ylabel('dFdr')
            plt.show()

            plt.figure()
            plt.scatter(rt_vec, dydr)
            plt.xlabel('r')
            plt.ylabel('dCpldr')
            plt.show()



