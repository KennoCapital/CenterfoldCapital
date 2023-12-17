import os
import pickle
import torch
import itertools
from matplotlib import ticker
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.products import CapletAsPutOnZCB
from application.engine.trolleSchwartz import trolleSchwartz
from application.utils.path_config import get_plot_path, get_data_path
from application.utils.torch_utils import max0
from application.engine.mcBase import mcSim, RNG, mcSimPaths
from joblib import Parallel, delayed

from scipy.interpolate import griddata
import numpy as np

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

MAX_PROCESSES = os.cpu_count() - 1
print(f'Settings MAX_PROCESSES = {MAX_PROCESSES}')


def get_training_data(X, t0, prd, N_train, const,
                      simDim: int = 1, seed: int = 1234, use_av: bool = True):
    # Calculate required sensitivities
    state = param_clone(*X)
    zcb, dudx = ZcbAAD().aad(
        *state, (prd.exerciseDate + prd.delta - t0).view(1), N_train, *const, simDim=simDim
    )

    state = param_clone(*X)
    payoff, dydx = CplAAD().aad(
        *state, t0, prd, N_train, *const, simDim=simDim, seed=seed, use_av=use_av
    )
    dydu = row_wise_chain_rule(dydx, dudx)

    # Format training data
    x_train = torch.hstack([zcb.reshape(N_train, 1), X[1].reshape(-1, 1)])
    y_train = payoff.reshape(N_train, 1)
    z_train = torch.hstack([dydu, dydx[:, 1].reshape(N_train, 1)])

    return x_train, y_train, z_train


def diff_reg_fit_predict(x_train, y_train, z_train, x_test, diff_reg, scalar):
    x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)
    diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)
    x_test_scaled, _, _ = scalar.transform(x_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(x_test_scaled, predict_derivs=True)
    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    return y_pred, z_pred


def row_wise_chain_rule(dydx, dudx):
    """
        1-Dimensional case of Linderstrøm eq. (3.22), i.e. 1 underlying.
        Arguments are assumed to be matrices where each observation is stored in a row.

        dydx is the derivative of the product (or payoff) wrt. the model parameters
        dudx is the derivative of the underlying wrt. the model parameters

        returns the derivative of the product (payoff) wrt. the underlying
    """
    num_features = dydx.shape[1]

    solve_rowwise = lambda a, b: torch.pinverse(a.T) @ b.T
    equations = (
        (dudx[i, :].reshape(-1, num_features), dydx[i, :].reshape(-1, num_features))
        for i in range(dydx.shape[0])
    )
    solutions = itertools.starmap(solve_rowwise, equations)
    dydu = torch.vstack(list(solutions))
    return dydu


def param_clone(xt, vt, phi1t, phi2t, phi3t, phi4t, phi5t, phi6t):
    x = xt.clone()
    v = vt.clone()
    phi1 = phi1t.clone()
    phi2 = phi2t.clone()
    phi3 = phi3t.clone()
    phi4 = phi4t.clone()
    phi5 = phi5t.clone()
    phi6 = phi6t.clone()
    return x, v, phi1, phi2, phi3, phi4, phi5, phi6


class CplAAD(torch.nn.Module):
    def __init__(self):
        super(CplAAD, self).__init__()

    def forward(self,
                xt, vt, phi1t, phi2t, phi3t, phi4t, phi5t, phi6t, t0, prd, N,
                kappa, theta, rho, sigma, alpha0, alpha1, gamma, varphi,
                simDim: int = 1, seed: int = 1234, use_av: bool = True):

        cMdl = trolleSchwartz(
            xt=xt, vt=vt, phi1t=phi1t, phi2t=phi2t, phi3t=phi3t, phi4t=phi4t, phi5t=phi5t, phi6t=phi6t,
            kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
            simDim=simDim
        )

        cPrd = CapletAsPutOnZCB(
            strike=prd.strike,
            exerciseDate=prd.exerciseDate - t0,
            delta=prd.delta,
            notional=prd.notional
        )

        cTL = dTL[dTL <= prd.exerciseDate - t0]
        cRng = RNG(seed=seed, use_av=use_av)
        payoffs = mcSim(cPrd, cMdl, cRng, N, cTL)

        return payoffs

    def aad(self,
            x, v, phi1, phi2, phi3, phi4, phi5, phi6, t0, prd, N,
            kappa, theta, rho, sigma, alpha0, alpha1, gamma, varphi,
            simDim: int = 1, seed: int = 1234, use_av: bool = True):

        idx_half = N
        if use_av:
            x = torch.hstack([x, x])
            v = torch.hstack([v, v])
            phi1 = torch.hstack([phi1, phi1])
            phi2 = torch.hstack([phi2, phi2])
            phi3 = torch.hstack([phi3, phi3])
            phi4 = torch.hstack([phi4, phi4])
            phi5 = torch.hstack([phi5, phi5])
            phi6 = torch.hstack([phi6, phi6])
            N *= 2

        x.requires_grad = True
        v.requires_grad = True
        phi1.requires_grad = True
        phi2.requires_grad = True
        phi3.requires_grad = True
        phi4.requires_grad = True
        phi5.requires_grad = True
        phi6.requires_grad = True

        payoff = self.forward(
            x, v, phi1, phi2, phi3, phi4, phi5, phi6, t0, prd, N,
            kappa, theta, rho, sigma, alpha0, alpha1, gamma, varphi,
            simDim, seed, use_av)
        payoff.backward(torch.ones((N,)), retain_graph=False)
        grad = torch.vstack([x.grad, v.grad, phi1.grad, phi2.grad, phi3.grad, phi4.grad, phi5.grad, phi6.grad]).T

        if use_av:
            payoff = 0.5 * (payoff[:idx_half] + payoff[idx_half:])
            grad = 0.5 * (grad[:idx_half] + grad[idx_half:])

        return payoff.detach(), grad


class ZcbAAD(torch.nn.Module):
    def __init__(self):
        super(ZcbAAD, self).__init__()

    def forward(self,
                xt, vt, phi1t, phi2t, phi3t, phi4t, phi5t, phi6t, tenor,
                kappa, theta, rho, sigma, alpha0, alpha1, gamma, varphi,
                simDim: int = 1):
        cMdl = trolleSchwartz(
            xt=xt, vt=vt, phi1t=phi1t, phi2t=phi2t, phi3t=phi3t, phi4t=phi4t, phi5t=phi5t, phi6t=phi6t,
            kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
            simDim=simDim
        )

        zcb = mdl.calc_zcb(cMdl.x0, t=torch.tensor([0.0]), T=tenor)

        return zcb

    def aad(self,
            x, v, phi1, phi2, phi3, phi4, phi5, phi6, tenor, N,
            kappa, theta, rho, sigma, alpha0, alpha1, gamma, varphi,
            simDim: int = 1):
        x.requires_grad = True
        v.requires_grad = True
        phi1.requires_grad = True
        phi2.requires_grad = True
        phi3.requires_grad = True
        phi4.requires_grad = True
        phi5.requires_grad = True
        phi6.requires_grad = True

        zcb = self.forward(
            x, v, phi1, phi2, phi3, phi4, phi5, phi6, tenor,
            kappa, theta, rho, sigma, alpha0, alpha1, gamma, varphi,
            simDim)
        zcb.backward(torch.ones((simDim, len(tenor), N,)), retain_graph=False)

        v.grad = torch.zeros_like(x.grad) if v.grad is None else v.grad
        grad = torch.vstack([x.grad, v.grad, phi1.grad, phi2.grad, phi3.grad, phi4.grad, phi5.grad, phi6.grad]).T

        return zcb.detach(), grad


if __name__ == '__main__':
    # Settings
    file_path = get_data_path('ts_cpl_mc_prices_diffML_withAV_states_training.pkl')
    # burn_in_dTL = torch.linspace(0.0, 1, 101)
    plot_nodes = 100

    seed = 42069
    N_train = 1024*8
    N_test = 256
    steps_per_year = 100
    use_av = True

    # Differential Regressor and Scalar
    deg = 5
    alpha = 1.0
    include_interactions = True

    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True,
                                                   include_interactions=include_interactions)
    scalar = DifferentialStandardScaler()

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = torch.tensor(0.07)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    # Monte Carlo
    rng = RNG(seed=seed, use_av=use_av)
    dTL = torch.linspace(0.0, float(prd.exerciseDate), int(steps_per_year * prd.exerciseDate) + 1)

    # Model specification
    simDim = 1
    xt = torch.tensor([0.0])
    vt = torch.tensor([0.0194])  * 5
    phi1t = torch.tensor([0.0])
    phi2t = torch.tensor([0.0])
    phi3t = torch.tensor([0.0])
    phi4t = torch.tensor([0.0])
    phi5t = torch.tensor([0.0])
    phi6t = torch.tensor([0.0])

    kappa = torch.tensor(0.0553)
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476) * 5
    sigma = torch.tensor(0.3325)
    rho = torch.tensor(0.4615)

    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)

    varphi = torch.tensor(0.0832)

    const = kappa, theta, sigma, rho, alpha0, alpha1, gamma, varphi

    mdl = trolleSchwartz(
        xt=xt, vt=vt, phi1t=phi1t, phi2t=phi2t, phi3t=phi3t, phi4t=phi4t, phi5t=phi5t, phi6t=phi6t,
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
        simDim=simDim
    )

    burn_in_dTL = torch.linspace(0.0, 1., 51)
    mcSimPaths(prd, mdl, rng, N_train, burn_in_dTL)

    t0 = torch.tensor(0.0)

    X = [x[:simDim, -1] for x in mdl.x]
    state0 = param_clone(*X)
    zcb, dudx = ZcbAAD().aad(
        *state0, (prd.exerciseDate + prd.delta - t0).view(1), N_train, *const, simDim=simDim
    )

    # compute training set for caplet
    state0 = param_clone(*X)
    payoff, dydx = CplAAD().aad(
        *state0, t0, prd, N_train, *const, simDim=simDim, seed=seed, use_av=use_av
    )
    dydu = row_wise_chain_rule(dydx, dudx)

    # X: (ZCB, v)
    X_train = torch.hstack((zcb.reshape(-1,1), X[1].reshape(-1,1))) # size N x 2

    # y: (cpl)
    y_train = payoff.reshape(-1,1) # size N x 1

    # Z: (dC/dP, dc/dv)
    Z_train = torch.hstack((dydu.reshape(-1,1), dydx[:,1].reshape(-1,1))) # size N x 2


    q = X_train.quantile(0.9, dim=0)
    cond = torch.prod(X_train <= q, dim=1).bool()
    idx_test = torch.randperm(len(X_train[cond]), generator=rng.gen)[:N_test]
    idx_test = idx_test.sort().values
    X_test = X_train[idx_test,:]

    # Fit and predict
    X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, Z_train)
    diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)
    X_test_scaled, _, _ = scalar.transform(X_test, None, None)
    # Predict
    y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)
    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    # MC price
    price = torch.zeros_like(y_pred)
    x_mc = torch.stack(state0)
    x_mc = x_mc[:, :, idx_test]
    for i in tqdm(range(N_test), desc='Calculating initial prices with MC'):
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [j for j in x_mc[:, :, i]]
        cMdl = trolleSchwartz(xt=x, vt=v, phi1t=phi1, phi2t=phi2, phi3t=phi3,
                              phi4t=phi4, phi5t=phi5, phi6t=phi6)
        cRng = RNG(seed=seed, use_av=use_av)
        payoff = mcSim(prd, cMdl, cRng, 50000, dTL)
        price[i] = torch.mean(payoff)

    RMSE_value = torch.sqrt(torch.mean((y_pred - price) ** 2))

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(price, price, '--', color='black', label='MC Price')
    ax.plot(price, y_pred.flatten(), 'o', color='orange', alpha=0.25, label=f'DiffReg (RMSE = {RMSE_value:.2f})')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_xlabel('MC Price')
    ax.grid(lw=0.5)
    ax.set_title('Price predictions of Caplet with Differential Regression')
    plt.show()

    # Make data used for plotting
    X_test2 = torch.tensor([
        (x, y)
        for x in torch.linspace(X_test[:, 0].quantile(0.0), X_test[:, 0].max(), plot_nodes)
        for y in torch.linspace(X_test[:, 1].quantile(0.0), X_test[:, 1].max(), plot_nodes)
    ])

    X_test2_scaled, _, _ = scalar.transform(X_test2, None, None)
    # Predict
    y_pred2_scaled, z_pred2_scaled = diff_reg.predict(X_test2_scaled, predict_derivs=True)
    _, y_pred2, z_pred2 = scalar.predict(None, y_pred2_scaled, z_pred2_scaled)

    # Reshape to grids for plotting
    x1_grid = X_test2[:, 0].reshape(plot_nodes, plot_nodes)
    x2_grid = X_test2[:, 1].reshape(plot_nodes, plot_nodes)
    y_grid = y_pred2.reshape(plot_nodes, plot_nodes)
    z1_grid = z_pred2[:, 0].reshape(plot_nodes, plot_nodes)
    z2_grid = z_pred2[:, 1].reshape(plot_nodes, plot_nodes)

    test_shape = tuple([int(torch.sqrt(torch.tensor(N_test))), int(torch.sqrt(torch.tensor(N_test)))])
    x1_grid_test = X_test[:, 0].reshape(test_shape)
    x2_grid_test = X_test[:, 1].reshape(test_shape)
    y_grid_test = price.reshape(test_shape)

    # Price
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf_pred = ax.plot_surface(x1_grid, x2_grid, y_grid, cmap=plt.cm.magma, linewidth=0, antialiased=False, alpha=0.8)
    scatter_test = ax.scatter(X_test[:, 0], X_test[:, 1], y_grid_test, alpha=1.0, color='black')
    ax.set_xlabel(r'$ZCB$', labelpad=10)
    ax.set_ylabel(r'$\nu$')
    ax.set_zticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    cbar = fig.colorbar(surf_pred, shrink=0.5, aspect=5, format=ticker.EngFormatter())
    cbar.ax.set_title('Price')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # ax.view_init(elev=20, azim=-100)
    ax.view_init(elev=30, azim=-60)
    # plt.savefig(path_name.format('price'), dpi=400)
    plt.show()

    # Delta
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf_pred = ax.plot_surface(x1_grid, x2_grid, z1_grid, cmap=plt.cm.magma)
    ax.set_xlabel(r'$ZCB$', labelpad=10)
    ax.set_ylabel(r'$\nu$')
    ax.set_zticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    # ax.xaxis.set_major_formatter(ticker.EngFormatter())
    cbar = fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    cbar.ax.set_title('Delta')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=30, azim=-110)
    # plt.savefig(path_name.format('delta'), dpi=400)
    plt.show()

    # Vega
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf_pred = ax.plot_surface(x1_grid, x2_grid, z2_grid, cmap=plt.cm.magma)
    ax.set_xlabel(r'$ZCB$', labelpad=10)
    ax.set_ylabel(r'$\nu$')
    ax.set_zticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    # ax.xaxis.set_major_formatter(ticker.EngFormatter())
    cbar = fig.colorbar(surf_pred, shrink=0.5, aspect=5, format=ticker.EngFormatter())
    cbar.ax.set_title('Vega')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=30, azim=-110)
    # plt.savefig(path_name.format('vega'), dpi=400)
    plt.show()

