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
import torch
import torch.nn as nn
from torch.func import jacrev, jacfwd
import pickle
import os

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 256*4
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
    exerciseDate = torch.tensor(5.0)
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
    p_vec = torch.exp(-varphi_grid * (exerciseDate + delta - t1))

    class payoffModule(nn.Module):
        def __init__(self, strike, exerciseDate, delta, notional, gamma, kappa, theta, rho, sigma, alpha0, alpha1):
            super(payoffModule, self).__init__()

            self.strike = strike
            self.exerciseDate = exerciseDate
            self.delta = delta
            self.notional = notional
            self.gamma = gamma
            self.kappa = kappa
            self.theta = theta
            self.rho = rho
            self.sigma = sigma
            self.alpha0 = alpha0
            self.alpha1 = alpha1

        def forward(self, vt_vec, p_vec, t1):
            t1 = torch.tensor(t1)

            if vt_vec.dim() == 1:
                vt_vec = vt_vec.reshape(1, -1)

            def _payoffs_v(vt_vec, p_vec):
                cvarphi = -torch.log(p_vec) / (self.exerciseDate + self.delta)
                cMdl = trolleSchwartz(xt=xt_vec[0, :, :], vt=vt_vec,
                                      phi1t=xt_vec[2, :, :], phi2t=xt_vec[3, :, :],
                                      phi3t=xt_vec[4, :, :], phi4t=xt_vec[5, :, :],
                                      phi5t=xt_vec[6, :, :], phi6t=xt_vec[7, :, :],
                                      gamma=self.gamma, kappa=self.kappa, theta=self.theta,
                                      rho=self.rho, sigma=self.sigma, alpha0=self.alpha0,
                                      alpha1=self.alpha1, varphi=cvarphi, simDim=1)
                cPrd = CapletAsPutOnZCB(
                    strike=self.strike,
                    exerciseDate=self.exerciseDate - t1,
                    delta=self.delta,
                    notional=self.notional
                )
                cTL = dTL[dTL <= self.exerciseDate - t1]
                payoffs = mcSim(cPrd, cMdl, rng, vt_vec.shape[-1], cTL)  # this is the slow part.
                return payoffs

            return _payoffs_v(vt_vec, p_vec)


    vt_vec = torch.tensor(vt_vec, requires_grad=True)
    p_vec = torch.tensor(p_vec, requires_grad=True)
    payoff = payoffModule(strike, exerciseDate, delta, notional, gamma, kappa, theta, rho, sigma, alpha0, alpha1)

    cpl = payoff(vt_vec, p_vec, t1)
    m = torch.ones(N_train)
    cpl.backward(m)

    """
    def calc_dcpl_dv(vt_vec, p_vec, t1):

        t1 = torch.tensor(t1)

        vt_vec = vt_vec.detach()
        p_vec = p_vec.detach()

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

    # c, dc = calc_dcpl_dv(vt_vec, p_vec, t1)

    
    x_train = torch.hstack([p_vec.reshape(-1,1), vt_vec.reshape(-1,1)])
    y_train = c
    z_train = dc
    """

    # Setup Differential Regressor, and Scalar
    deg = 5
    alpha = 1.0
    include_interactions = True
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True,
                                               include_interactions=include_interactions)
    scalar = DifferentialStandardScaler()


    p, v = p_vec.detach(), vt_vec.detach()
    p_grad = p_vec.grad.detach()
    v_grad = vt_vec.grad.detach()

    x_train = torch.hstack([p.reshape(-1,1), v.reshape(-1,1)])
    y_train = cpl.detach().reshape(-1,1)
    z_train = torch.hstack([p_grad.reshape(-1,1), v_grad.reshape(-1,1)])

    zcb_test = torch.linspace(p.min(), p.max(), 32)
    v0_test = torch.linspace(torch.abs(v).min(), v.max(), 32)
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
    z_delta = z_pred[:, 0].reshape(32, 32) / notional
    z_vega = z_pred[:, 1].reshape(32, 32) / notional


    # Price
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    surf_pred = ax.plot_surface(x_, y_, z_price, cmap=plt.cm.YlOrBr)


    ax.set_xlabel('zcb')
    ax.set_ylabel('v')
    ax.set_zlabel('Payoff')

    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()

    # Delta
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    surf_pred = ax.plot_surface(x_, y_, z_delta, cmap=plt.cm.rainbow)

    ax.set_xlabel('zcb')
    ax.set_ylabel('v')
    ax.set_zlabel('Delta')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()

    # Vega
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    surf_pred = ax.plot_surface(x_, y_, z_vega, cmap=plt.cm.rainbow)

    ax.set_xlabel('zcb')
    ax.set_ylabel('v')
    ax.set_zlabel('vega')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()


