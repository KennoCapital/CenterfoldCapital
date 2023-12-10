import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.experiments.trolle_schwartz.ts_hedge_tools import training_data
from application.engine.products import CapletAsPutOnZCB
from application.engine.trolleSchwartz import trolleSchwartz
from application.utils.path_config import get_plot_path, get_data_path
from application.engine.mcBase import mcSim, RNG
from torch.func import jacrev, jacfwd
import pickle
import os

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 128
    N_test = 128
    use_av = True
    save_plot = False

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553)
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476)
    v0 = theta

    # initializer
    #varphi_min = 0.03
    #varphi_max = 0.13
    #varphi = torch.linspace(varphi_min, varphi_max, N_train)

    varphi = torch.tensor(0.0832)

    model = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = torch.tensor(.07)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    """ Helper functions for generating training data of pathwise payoffs and deltas """


    def calc_dzcb_dv(v0_vec, t0):
        """
        :param  x0_vec:    Possible state variables of state space
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)

        #v0_vec = torch.tensor(v0_vec, requires_grad=True)

        v0_vec = torch.tensor(v0_vec, requires_grad=True)

        def _zcb(v0_vec):
            tmp_mdl = trolleSchwartz(v0_vec, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            mcSim(prd, tmp_mdl, rng, len(v0_vec), dTL)
            state = torch.stack(tmp_mdl.x)[:, :, 0, :]
            zcb = model.calc_zcb(state, t0, exerciseDate + delta)[0]
            return zcb

        zcbs = _zcb(v0_vec) # size N

        J = jacfwd(_zcb, argnums=(0), randomness='same')(v0_vec)
        return zcbs, J.sum_to_size(len(v0_vec))


    def calc_dcpl_dv(v0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)
        v0_vec = torch.tensor(v0_vec, requires_grad=True)

        def _payoffs(v0_vec):
            cMdl = trolleSchwartz(v0_vec, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(v0_vec), cTL)
            return payoffs

        cpls = _payoffs(v0_vec)

        J = jacfwd(_payoffs, argnums=(0), randomness='same')(v0_vec)

        return cpls.detach(), J.sum_to_size(len(v0_vec)).detach()

    def calc_dCdP(zcbs):
        zcbs = torch.tensor(zcbs, requires_grad=True)
        def _payoffs(zcbs,v0=v0):

            varphi = -torch.log(zcbs) / (exerciseDate+delta)

            cMdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(varphi), cTL)
            return payoffs

        cpls = _payoffs(zcbs)
        J = jacfwd(_payoffs, argnums=(0), randomness='same')(zcbs)

        return cpls.detach(), J.sum_to_size(len(zcbs)).detach()


    def calc_dXdv(v0):

        def _x(v0):
            tmp_mdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            mcSim(prd, tmp_mdl, rng, N_train, dTL)
            x, v, phi1, phi2, phi3, phi4, phi5, phi6 = tmp_mdl.x

            # torch.stack(tmp_mdl.x)[:, :, -1, :]
            return (x, v, phi1, phi2, phi3, phi4, phi5, phi6)

        v0 = torch.tensor(v0, requires_grad=True)
        dvecs = jacfwd(_x, has_aux=False, randomness='same')(v0)  # list of state vars sens. to v0
        jac = torch.stack(dvecs)

        return None


    hedge_points = 50
    dTL = torch.linspace(0.0, exerciseDate, int(hedge_points + 1))

    """ Calculate  sens """
    t0 = torch.tensor(0.)
    v0_vec = torch.linspace(0.01, 0.1, N_train)

    """
    if use_av:
        v0_vec = torch.concat([v0_vec, v0_vec], dim=0)
        model._v0 = v0_vec
    """

    X_train, y_train, z_train = training_data(x0_vec=v0_vec,
                                              t0=0.0,
                                              calc_dPrd_dr=calc_dcpl_dv,
                                              calc_dU_dr=calc_dzcb_dv,
                                              use_av=use_av)

    X_train = X_train.detach()
    y_train = y_train.detach()
    z_train = z_train.detach()


    plt.figure()
    plt.plot(X_train, y_train, 'o')
    plt.show()

    C, dC = calc_dcpl_dv(v0_vec, t0)



    y_mdl = torch.full_like(v0_vec, torch.nan)
    for j in tqdm(range(len(v0_vec))):
        tmp_mdl = trolleSchwartz(v0_vec[j], gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
        tmp_rng = RNG(seed=seed, use_av=use_av)
        y_mdl[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 10000, dTL)))
    y_mdl = y_mdl.reshape(-1, 1)
    z_mdl = y_mdl.diff(dim=0) / v0_vec.diff(dim=0)

    plt.figure()
    plt.plot(v0_vec, dC.detach().numpy() / notional ,'o' )
    plt.plot(v0_vec[1:], z_mdl/notional, 'o')
    plt.xlabel(r'$v0$')
    plt.ylabel(r'$dC / dv_0 $')
    plt.show()

    X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

    diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

    X_test = X_train.detach()
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
    if save_plot:
        plt.savefig(get_plot_path('ts_AAD_DiffReg_plot_delta_caplet_1D.png'), dpi=400)
    plt.show()



