import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.experiments.trolle_schwartz.ts_hedge_tools import training_data
from application.engine.products import CapletAsPutOnZCB
from application.engine.trolleSchwartz import trolleSchwartz
from application.utils.path_config import get_plot_path
from application.engine.mcBase import mcSim, RNG
from torch.func import jacrev

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 256*4
    N_test = 256
    use_av = False

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=True)
    scalar = DifferentialStandardScaler()

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553) #0553
    sigma = torch.tensor(0.3325) #3325
    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476)

    # initializer
    varphi_min = 0.07
    varphi_max = 0.11
    varphi = torch.linspace(varphi_min, varphi_max, N_train)
    #varphi = torch.empty(N_train) #torch.tensor(0.0832)
    #varphi = varphi.fill_(0.0832)

    # only chosen for time-0
    """
    init = torch.empty(N_train)
    init = init.fill_(varphi)
    """

    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = torch.tensor(.087)
    #strike = mdl.calc_swap_rate(r0, exerciseDate, delta)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    """ Helper functions for generating training data of pathwise payoffs and deltas """
    def calc_dzcb_dx(x0_vec, t0):
        """
        :param  x0_vec:    Possible state variables of state space
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)

        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [x.reshape(1, -1) for x in x0_vec]

        def _zcb(x, v, phi1, phi2, phi3, phi4, phi5, phi6):
            state = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
            zcb = model.calc_zcb(state, t0, exerciseDate + delta)[0]
            return zcb

        zcbs = _zcb(x, v, phi1, phi2, phi3, phi4, phi5, phi6)

        jac = jacrev(_zcb, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((1, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        dzcbs = tmp.unsqueeze(dim=2)

        return zcbs, dzcbs


    def calc_dcpl_dx(x0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [x.reshape(1, -1) for x in x0_vec]

        def _payoffs(x, v, phi1, phi2, phi3, phi4, phi5, phi6):
            state = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
            f0T = model.calc_instant_fwd(state, t0, exerciseDate + delta - t0)
            cMdl = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, f0T)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(f0T), cTL)
            return payoffs[0, :]

        cpls = _payoffs(x, v, phi1, phi2, phi3, phi4, phi5, phi6)

        jac = jacrev(_payoffs, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((1, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        dcpls = tmp.unsqueeze(dim=2)

        return cpls, dcpls

    """ Calculate `true` caplet price using Monte Carlo for comparison """
    #X_test = torch.exp(-varphi * exerciseDate).reshape(-1, 1)

    hedge_points = 5
    #dTL = torch.linspace(0.0, float(exerciseDate+delta), hedge_points + 1)
    dTL = torch.linspace(0.0, exerciseDate + delta, int(hedge_points * (exerciseDate + delta) + 1))

    mcSim(prd, model, rng, N_train, dTL).reshape(-1, 1)
    x0_vec = torch.stack(model.x)[:, :, -1, :]

    # Required when using AV to concat the initial forward rates
    if use_av:
        varphi = torch.concat([varphi, varphi], dim=0)
        model.varphi = varphi

    """ Estimate Price and Delta using Differential Regression """

    X_train, y_train, z_train = training_data(x0_vec=x0_vec,
                                              t0=0.0,
                                              calc_dPrd_dr=calc_dcpl_dx,
                                              calc_dU_dr=calc_dzcb_dx,
                                              use_av=use_av)

    plt.figure()
    plt.plot(X_train, z_train, 'o')
    plt.show()


    #X_test = model.calc_zcb(x0_vec, torch.tensor(0.0), exerciseDate + delta).reshape(-1, 1)
    X_test = torch.linspace(max(X_train.min(),X_train.mean()-3*X_train.std() ), min(X_train.max(), X_train.mean()+3*X_train.std()), N_test).reshape(-1, 1)

    varphi = -torch.log(X_test)/(exerciseDate + delta)  # torch.tensor(0.0832)
    y_mdl = torch.full_like(X_test, torch.nan)
    for j in tqdm(range(len(X_test))):
        tmp_mdl = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi[j])
        tmp_rng = RNG(seed=seed, use_av=use_av)
        y_mdl[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 10000, dTL)))
    y_mdl = y_mdl.reshape(-1, 1)
    z_mdl = y_mdl.diff(dim=0) / X_test.diff(dim=0)

    X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

    diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

    X_test_scaled, _, _ = scalar.transform(X_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    RMSE_price = torch.sqrt(torch.mean((y_pred - y_mdl) ** 2))
    MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_mdl))

    """ Plot results """

    fig, ax = plt.subplots(2, sharex='col')
    # Plot price function
    ax[0].plot(X_train.flatten(), y_train.flatten(), 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[0].plot(X_test.flatten(), y_pred, label='DiffReg', color='orange')
    ax[0].plot(X_test, y_mdl, color='black', label='MC (Bump and reval)')
    ax[0].set_ylabel('Price')
    ax[0].text(0.05, 0.8, f'RMSE = {RMSE_price:.2f}', fontsize=8, transform=ax[0].transAxes)

    # Plot delta function
    ax[1].plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[1].plot(X_test, z_pred, label='DiffReg', color='orange')
    ax[1].plot(X_test[1:], z_mdl, color='black', label='MC (Bump and reval)')
    ax[1].set_xlabel('Swap(0)')
    ax[1].set_ylabel('Delta')
    ax[1].text(0.05, 0.8, f'MAE = {MAE_delta:.4f}', fontsize=8, transform=ax[1].transAxes)

    # Adjust size of subplots
    box0 = ax[0].get_position()
    ax[0].set_position([box0.x0, box0.y0 - box0.height * 0.1, box0.width, box0.height * 0.9])

    box1 = ax[1].get_position()
    ax[1].set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))

    # Title
    av_str = 'with AV' if use_av else 'without AV'
    fig.suptitle(prd.name + f'\nalpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str)

    # plt.savefig(get_plot_path('vasicek_AAD_DiffReg_EuSwpt_closetoT_Naive.png'), dpi=400)
    plt.show()

