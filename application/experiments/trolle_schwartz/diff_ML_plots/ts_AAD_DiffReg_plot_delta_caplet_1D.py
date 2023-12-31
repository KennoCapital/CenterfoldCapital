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
    N_train = 256
    N_test = 256
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
    varphi = torch.linspace(varphi_min, varphi_max, N_train)
    #varphi = torch.ones(N_train) * 0.0832

    """
    model = trolleSchwartz(vt=v0, gamma=gamma, kappa=kappa, theta=theta, rho=rho,
                           sigma=sigma, alpha0=alpha0, alpha1=alpha1, varphi=varphi, simDim=1)
    """

    model = trolleSchwartz(xt=torch.tensor([0.0]),
                 phi1t=torch.tensor([0.0]),
                 phi2t=torch.tensor([0.]),
                 phi3t=torch.tensor([0.]),
                 phi4t=torch.tensor([0.]),
                 phi5t=torch.tensor([0.]),
                 phi6t=torch.tensor([0.]),
                 vt=v0, gamma=gamma, kappa=kappa, theta=theta, rho=rho,
                sigma=sigma, alpha0=alpha0, alpha1=alpha1, varphi=varphi, simDim=1)


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
    def calc_dzcb_dx(x0_vec, t0):
        """
        :param  x0_vec:    Possible state variables of state space
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)

        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [x.reshape(model.simDim, -1) for x in x0_vec]

        def _zcb(x, v, phi1, phi2, phi3, phi4, phi5, phi6):
            state = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
            zcb = model.calc_zcb(state, t0, exerciseDate + delta)[0]
            return zcb

        zcbs = _zcb(x, v, phi1, phi2, phi3, phi4, phi5, phi6) # size N

        # first we find x, phis state var sens (since the method 'calc_zcb' will not give us any sens wrt v)
        # the reverse mode is efficient for R^n -> R^m when n>m
        jac = jacrev(_zcb, argnums=(0, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((model.simDim, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        tmp = tmp.unsqueeze(dim=2) # size N x 7 (!) x simDim

        # to track v we have to "show" auto diff how v is influencing other state vars
        #v = torch.tensor(v, requires_grad=True)
        def _zcb_v(v):
            tmp_mdl = trolleSchwartz(v, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            mcSim(prd, tmp_mdl, rng, len(varphi), dTL[::2]) # this is the slow part. Now skipping every other
            state = torch.stack(tmp_mdl.x)[:, :, int(t0), :]
            zcb = model.calc_zcb(state, t0, exerciseDate + delta)[0]
            return zcb

        # forward mode when just one argument
        J = jacfwd(_zcb_v, argnums=(0), randomness='same')(v).detach()
        J = J.sum_to_size(len(varphi)) # size N

        # **we append dP/dv in the last column**
        dzcbs = torch.cat( (tmp, J.reshape(-1,1,1)), dim=1) # size N x 8 x simDim

        return zcbs[0], dzcbs


    def calc_dcpl_dx(x0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [x.reshape(model.simDim, -1) for x in x0_vec]
        def _payoffs(x, v, phi1, phi2, phi3, phi4, phi5, phi6):
            state = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
            f0T = model.calc_instant_fwd(state, t0, exerciseDate + delta - t0).flatten()
            cMdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, f0T, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(f0T), cTL)
            return payoffs

        cpls = _payoffs(x, v, phi1, phi2, phi3, phi4, phi5, phi6)

        # the reverse mode is efficient for R^n -> R^m when n>m
        jac = jacfwd(_payoffs, argnums=(0, 2, 3, 4, 5, 6, 7), randomness='same')(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        #jac = jacrev(_payoffs, argnums=(0, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((1, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        tmp = tmp.unsqueeze(dim=2) # size N x 7 (!) x simDim

        #v = torch.tensor(v, requires_grad=True)
        def _payoffs_v(v):
            cMdl = trolleSchwartz(v, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(varphi), cTL[::2]) # this is the slow part. Now skipping every other
            return payoffs

        J = jacfwd(_payoffs_v, argnums=(0), randomness='same')(v).detach()
        J = J.sum_to_size(len(varphi))

        # **we append dP/dv in the last column**
        dcpls = torch.cat((tmp, J.reshape(-1, 1, 1)), dim=1)  # size N x 8 x simDim

        return cpls, dcpls

    """ Calculate `true` caplet price using Monte Carlo for comparison """
    #X_test = torch.exp(-varphi * exerciseDate).reshape(-1, 1)

    hedge_points = 100
    #dTL = torch.linspace(0.0, float(exerciseDate+delta), hedge_points + 1)
    dTL = torch.linspace(0.0, exerciseDate, int(hedge_points * exerciseDate + 1))

    mcSim(prd, model, rng, N_train, dTL)
    x0_vec = torch.stack(model.x)[:, :, 0, :]


    p, dp = calc_dzcb_dx(x0_vec, t0=0.)
    c, dc = calc_dcpl_dx(x0_vec, t0=0.)

    plt.figure()
    plt.plot(p, dp[:,0,0], 'o', label='dx')
    plt.plot(p, dp[:, 1, 0] , 'o', label='dphi1')
    plt.plot(p, dp[:, 2, 0] / 1., 'o', label='dphi2')
    plt.plot(p, dp[:, 3, 0] / 1, 'o', label='dphi3')
    plt.plot(p, dp[:, 4, 0] / 1, 'o', label='dphi4')
    plt.plot(p, dp[:, 5, 0] / 1, 'o', label='dphi5')
    plt.plot(p, dp[:, 6, 0] / 1, 'o', label='dphi6')
    plt.plot(p, dp[:, 7, 0] / 1, 'o', label='dv', alpha=0.2)
    plt.legend()
    plt.xlabel('P(0,T)')
    plt.ylabel('dP(0,T)')
    plt.show()

    plt.figure()
    plt.plot(p, dc[:, 0, 0] / notional, 'o', label='dx')
    plt.plot(p, dc[:, 1, 0] / notional, 'o', label='dphi1')
    plt.plot(p, dc[:, 2, 0] / notional, 'o', label='dphi2')
    plt.plot(p, dc[:, 3, 0] / notional, 'o', label='dphi3')
    plt.plot(p, dc[:, 4, 0] / notional, 'o', label='dphi4')
    plt.plot(p, dc[:, 5, 0] / notional, 'o', label='dphi5')
    plt.plot(p, dc[:, 6, 0] / notional, 'o', label='dphi6')
    plt.plot(p, dc[:, 7, 0] / notional, 'o', label='dv', alpha=0.2)
    plt.legend()
    #plt.ylim(-0.1, 0.1)
    plt.xlabel('P(0,T)')
    plt.ylabel('dcpl')
    plt.show()



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
    plt.plot(X_train, y_train, 'o')
    plt.show()

    plt.figure()
    plt.plot(X_train, z_train, 'o')
    plt.show()

    X_test = torch.linspace(max(X_train.min(),X_train.mean()-5*X_train.std() ), min(X_train.max(), X_train.mean()+5*X_train.std()), N_test).reshape(-1, 1)

    varphi = -torch.log(X_test)/(exerciseDate + delta)
    y_mdl = torch.full_like(X_test, torch.nan)
    for j in tqdm(range(len(X_test))):
        tmp_mdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi[j], simDim=1)
        tmp_rng = RNG(seed=seed, use_av=use_av)
        y_mdl[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 10000//2, dTL)))
    y_mdl = y_mdl.reshape(-1, 1)
    z_mdl = y_mdl.diff(dim=0) / X_test.diff(dim=0)
    """
    filename = get_data_path('ts_mc_cpl_test_set.pkl')
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            y_mdl, z_mdl = pickle.load(file)
    else:
        #varphi = -torch.log(X_test)/(exerciseDate + delta)  # torch.tensor(0.0832)
        y_mdl = torch.full_like(X_test, torch.nan)
        for j in tqdm(range(len(X_test))):
            tmp_mdl = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi_red[j])
            tmp_rng = RNG(seed=seed, use_av=use_av)
            y_mdl[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 10000*10, dTL)))
        y_mdl = y_mdl.reshape(-1, 1)
        z_mdl = y_mdl.diff(dim=0) / X_test.diff(dim=0)

        with open(filename,'wb') as file:
            pickle.dump((y_mdl, z_mdl),file,pickle.HIGHEST_PROTOCOL)
    """
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
    ax[1].plot(X_train, z_train/notional, 'o', color='gray', alpha=0.25, label='Pathwise samples')
    ax[1].plot(X_test, z_pred/notional, label='DiffReg', color='orange')
    ax[1].plot(X_test[1:], z_mdl/notional, color='black', label='MC (Bump and reval)')
    ax[1].set_xlabel('P(0,T)')
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

