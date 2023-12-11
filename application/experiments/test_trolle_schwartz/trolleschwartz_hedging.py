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
    varphi = torch.tensor(0.0832)

    # instantiate model
    model = trolleSchwartz(xt=torch.tensor([0.0]),vt=torch.tensor([0.0]),
                 phi1t=torch.tensor([0.0]),phi2t=torch.tensor([0.0]),
                 phi3t=torch.tensor([0.0]),phi4t=torch.tensor([0.0]),
                 phi5t=torch.tensor([0.0]),phi6t=torch.tensor([0.0]),
                           gamma=gamma, kappa=kappa, theta=theta,
                           rho=rho, sigma=sigma, alpha0=alpha0,
                           alpha1=alpha1, varphi=varphi, simDim=1)

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


    hedge_points = 100
    dTL = torch.linspace(0.0, exerciseDate, int(hedge_points * exerciseDate + 1))

    # burn in
    t1 = torch.tensor(0.25)
    burnTL = torch.linspace(0.0, t1, int(hedge_points * t1 + 1))
    mcSim(prd, model, rng, N_train, burnTL)
    xt_vec = torch.stack(model.x)[:, :, -1, :] # size 8 x 1 x N_train

    vt_vec = xt_vec[1,:,:]

    # zcb time-t1 grid
    varphi_grid = torch.linspace( varphi*0.5, varphi*1.5, N_train )
    p_vec = torch.exp(-varphi_grid * (exerciseDate + delta - t1)) # todo: consider init of this

    def calc_dcpl_dv(vt_vec, p_vec, t1):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        t1 = torch.tensor(t1)

        if p_vec.dim() == 1:
            p_vec = p_vec.reshape(1,-1)

        def _payoffs_v(vt_vec, p_vec):
            cvarphi = - torch.log(p_vec) / ( exerciseDate + delta)
            cMdl = trolleSchwartz(xt=xt_vec[0,:,:],vt=vt_vec,
                 phi1t=xt_vec[2,:,:],phi2t=xt_vec[3,:,:],
                 phi3t=xt_vec[4,:,:],phi4t=xt_vec[5,:,:],
                 phi5t=xt_vec[6,:,:],phi6t=xt_vec[7,:,:],
                           gamma=gamma, kappa=kappa, theta=theta,
                           rho=rho, sigma=sigma, alpha0=alpha0,
                           alpha1=alpha1, varphi=cvarphi, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t1,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t1]
            payoffs = mcSim(cPrd, cMdl, rng, vt_vec.shape[-1], cTL) # this is the slow part.
            return payoffs, payoffs

        J, C = jacfwd(_payoffs_v, argnums=(0, 1), has_aux=True, randomness='same')(vt_vec, p_vec)
        J = torch.stack( (J[0].sum_to_size(vt_vec.shape[-1]), J[1].sum_to_size(vt_vec.shape[-1])) )
        dCdv = J.sum_to_size(vt_vec.shape[-1])

        return C, dCdv



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

