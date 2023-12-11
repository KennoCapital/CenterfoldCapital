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
        dCdv = torch.stack( (J[0].sum_to_size(vt_vec.shape[-1]), J[1].sum_to_size(vt_vec.shape[-1])) )

        return C.reshape(-1,1), dCdv.reshape(-1,2)


        # Setup Differential Regressor, and Scalar
        deg = 5
        alpha = 1.0
        include_interactions = True
        diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True,
                                                   include_interactions=include_interactions)
        scalar = DifferentialStandardScaler()

        c, dc = calc_dcpl_dv(vt_vec, p_vec, t1)

        x_train = torch.hstack([p_vec.reshape(-1,1), vt_vec.reshape(-1,1)])
        y_train = c
        z_train = dc

        zcb_test = torch.linspace(p_vec.min(), p_vec.max(), 16)
        v0_test = torch.linspace(vt_vec.min(), vt_vec.max(), 16)
        x_test = torch.cartesian_prod(zcb_test, v0_test)

        x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)

        diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)

        x_test_scaled, _, _ = scalar.transform(x_test, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(x_test_scaled, predict_derivs=True)

        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        " Plot Results "
        # Data for plotting a surface
        x_ = x_test[:, 0].reshape(32, 32)
        y_ = x_test[:, 1].reshape(32, 32)
        z_price = y_pred.reshape(32, 32)
        z_delta = z_pred[:, 0].reshape(32, 32)
        z_vega = z_pred[:, 1].reshape(32, 32)
        zero = torch.zeros(32, 32)

        # Price
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
        surf_train = ax.scatter(zcb_test, v0, payoff, c='gray', alpha=1.0)
        # surf_pred = ax.scatter(x_, y_, z_, c=y_pred, cmap=plt.cm.magma)
        surf_pred = ax.plot_surface(x_, y_, z_price, cmap=plt.cm.magma)
        # surf_zero = ax.plot_surface(x_, y_, zero, color='blue', alpha=0.5)

        ax.set_xlabel('Swap(0)')
        ax.set_ylabel('v(0)')
        ax.set_zlabel('Payoff')

        fig.colorbar(surf_pred, shrink=0.5, aspect=5)
        plt.show()

        # Delta
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
        surf_train = ax.scatter(swap, v0, z_train[:, 0], c='gray', alpha=1.0)
        surf_pred = ax.plot_surface(x_, y_, z_delta, cmap=plt.cm.magma)

        ax.set_xlabel('Swap(0)')
        ax.set_ylabel('v(0)')
        ax.set_zlabel('Delta')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf_pred, shrink=0.5, aspect=5)
        plt.show()

        # Vega
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
        surf_train = ax.scatter(swap, v0, z_train[:, 1], c='gray', alpha=1.0)
        surf_pred = ax.plot_surface(x_, y_, z_vega, cmap=plt.cm.magma)

        ax.set_xlabel('Swap(0)')
        ax.set_ylabel('v(0)')
        ax.set_zlabel('Payoff')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf_pred, shrink=0.5, aspect=5)
        plt.show()


