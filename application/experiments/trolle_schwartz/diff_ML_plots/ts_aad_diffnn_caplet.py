import os
import pickle
import torch
import itertools
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
from application.engine.differential_NN import Neural_Approximator


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

    seed = 1234
    N_train = 1024*4
    N_test = 256
    steps_per_year = 100
    use_av = True

    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = 256 * 10
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4

    """
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True,
                                                   include_interactions=include_interactions)
    scalar = DifferentialStandardScaler()
    """

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
    xt = torch.tensor([0.0]) * torch.ones(N_train)
    vt = torch.tensor([0.1]) * torch.ones(N_train)
    phi1t = torch.tensor([0.0]) * torch.ones(N_train)
    phi2t = torch.tensor([0.0]) * torch.ones(N_train)
    phi3t = torch.tensor([0.0]) * torch.ones(N_train)
    phi4t = torch.tensor([0.0]) * torch.ones(N_train)
    phi5t = torch.tensor([0.0]) * torch.ones(N_train)
    phi6t = torch.tensor([0.0]) * torch.ones(N_train)

    kappa = torch.tensor(0.0553)
    theta = torch.tensor(.1) #7542* kappa / torch.tensor(2.1476)
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

    # Make predictions
    idx_test = torch.randperm(N_train, generator=rng.gen)[:N_test]
    X_test = X_train[idx_test, :]

    # Fit Network
    # Setup Differential Neutral Network
    diff_nn = Neural_Approximator(X_train, y_train, Z_train)
    diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                    hidden_layers=hidden_layers)
    diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
    y_pred, z_pred = diff_nn.predict_values_and_derivs(X_test)


    # MC price
    price = torch.zeros_like(y_pred)
    for i in tqdm(range(N_test), desc='Calculating initial prices with MC'):
        x_mc = torch.stack(state0)
        x_mc = x_mc[:,:,idx_test]
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [j for j in x_mc[:, :, i]]
        cMdl = trolleSchwartz(xt=x, vt=v, phi1t=phi1, phi2t=phi2, phi3t=phi3,
                              phi4t=phi4, phi5t=phi5, phi6t=phi6)
        cRng = RNG(seed=seed, use_av=use_av)
        payoff = mcSim(prd, cMdl, cRng, 50000, dTL)
        price[i] = torch.mean(payoff)

    z_mdl = price.flatten().diff(dim=0) / X_test[:,0].flatten().diff(dim=0)

    # Data for plotting a surface
    x_ = X_test[:, 0].reshape(16, 16) # zcb
    y_ = X_test[:, 1].reshape(16, 16) # vol
    z_price = y_pred.reshape(16, 16) # cpl price
    z_delta = z_pred[:, 0].reshape(16, 16) / notional # dc/dp
    z_vega = z_pred[:, 1].reshape(16, 16) / notional # dc/dv

    from scipy.interpolate import griddata
    import numpy as np
    x_grid, y_grid = np.meshgrid(np.linspace(x_.min(), x_.max(), 100), np.linspace(y_.min(), y_.max(), 100))
    z_price_grid = griddata((x_.flatten(), y_.flatten()), z_price.flatten(), (x_grid, y_grid), method='cubic')
    z_delta_grid = griddata((x_.flatten(), y_.flatten()), z_delta.flatten(), (x_grid, y_grid), method='cubic')
    z_vega_grid = griddata((x_.flatten(), y_.flatten()), z_vega.flatten(), (x_grid, y_grid), method='cubic')

    # Price
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf_pred = ax.plot_surface(x_grid, y_grid, z_price_grid, cmap=plt.cm.magma, linewidth=0, antialiased=False)
    scatter_train = ax.scatter(X_train[:, 0].reshape(64, 64), X_train[:, 1].reshape(64, 64), y_train.reshape(64, 64),
                               alpha=0.05, color='gray')
    ax.set_ylim(0., 0.5)
    ax.set_xlim(0.7, 1.05)
    ax.set_xlabel(r'$P(0,T+\delta)$')
    ax.set_ylabel(r'$\nu(0)$')
    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()

    # Delta
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    surf_pred = ax.plot_surface(x_grid, y_grid, z_delta_grid, cmap=plt.cm.magma)
    scatter_train = ax.scatter(X_train[:, 0].reshape(64, 64), X_train[:, 1].reshape(64, 64), Z_train[:,0].reshape(64, 64)/notional,
                               alpha=0.05, color='gray')
    ax.set_ylim(0., 0.5)
    ax.set_xlim(0.7, 1.05)
    ax.set_xlabel(r'$P(0,T+\delta)$')
    ax.set_ylabel(r'$\nu(0)$')
    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()

    # Vega
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    surf_pred = ax.plot_surface(x_grid, y_grid, z_vega_grid, cmap=plt.cm.magma)
    scatter_train = ax.scatter(X_train[:, 0].reshape(64, 64), X_train[:, 1].reshape(64, 64),
                               Z_train[:, 1].reshape(64, 64) / notional,
                               alpha=0.1, color='gray')
    ax.set_ylim(0., 0.5)
    ax.set_zlim(0, 0.025)
    ax.set_xlim(0.7, 1.05)
    ax.set_xlabel(r'$P(0,T+\delta)$')
    ax.set_ylabel(r'$\nu(0)$')
    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()


    plt.figure()
    plt.plot(zcb.flatten(), y_train, 'o', color='gray', alpha=0.2)
    plt.plot(X_test[:,0], y_pred, 'o', label='diff reg')
    plt.xlabel('zcb')
    plt.xlim(0.825, 1.)
    plt.legend()
    plt.show()

