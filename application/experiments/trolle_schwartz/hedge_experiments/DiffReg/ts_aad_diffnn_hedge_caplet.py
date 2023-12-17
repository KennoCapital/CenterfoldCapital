import torch
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from application.engine.products import CapletAsPutOnZCB
from application.engine.trolleSchwartz import trolleSchwartz
from application.utils.path_config import get_plot_path, get_data_path

from application.engine.mcBase import mcSim, RNG, mcSimPaths

from application.engine.differential_NN import Neural_Approximator


torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

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

def fit_predict_nn_delta_vega(state0, prd,):
    # compute zcb training set
    zcb, dudx = ZcbAAD().aad(
            *state0, (prd.exerciseDate + prd.delta - t0).view(1), N_train, *const, simDim=simDim
        )

    # compute training set for caplet
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
    diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam,
                    hidden_units=hidden_units, hidden_layers=hidden_layers)
    diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
    y_pred, z_pred = diff_nn.predict_values_and_derivs(X_test)
    return y_pred, z_pred


if __name__ == '__main__':
    # Settings
    file_path = get_data_path('ts_cpl_mc_prices_diffML_withAV_states_training.pkl')
    # burn_in_dTL = torch.linspace(0.0, 1, 101)

    seed = 1234
    N_train = 1024*16
    N_test = 256
    steps_per_year = 100
    use_av = True

    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = int(N_train * 5/8) #256 * 10
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4


    # Product specification
    exerciseDate = torch.tensor(1.)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike_sold = torch.tensor(0.07)
    strike_hedge = torch.tensor(0.09)

    prd_sold = CapletAsPutOnZCB(
        strike=strike_sold,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    prd_hedge = CapletAsPutOnZCB(
        strike=strike_hedge,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    # Monte Carlo
    rng = RNG(seed=seed, use_av=use_av)
    dTL = torch.linspace(0.0, float(prd_sold.exerciseDate), int(steps_per_year * prd_sold.exerciseDate) + 1)

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
    mcSimPaths(prd_sold, mdl, rng, N_train, burn_in_dTL)

    t0 = torch.tensor(0.0)

    X = [x[:simDim, -1] for x in mdl.x]
    state0 = param_clone(*X)


    # Select random states as test paths
    idx_test = torch.randperm(N_train, generator=rng.gen)[:N_test]
    X_test = [x[0, idx_test] for x in mdl.x0]

    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            price_sold, price_hedge = pickle.load(file)
    else:
        price_sold = torch.full((N_test,), torch.nan)
        price_hedge = torch.full((N_test,), torch.nan)
        for i in tqdm(range(N_test), desc='Calculating initial prices with MC'):
            x_mc = torch.stack(state0)
            x_mc = x_mc[:, :, idx_test]
            x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [j for j in x_mc[:, :, i]]
            cMdl = trolleSchwartz(xt=x, vt=v, phi1t=phi1, phi2t=phi2, phi3t=phi3,
                                  phi4t=phi4, phi5t=phi5, phi6t=phi6)
            cRng = RNG(seed=seed, use_av=use_av)
            paths = mcSimPaths(prd_sold, cMdl, cRng, 10000, dTL)
            price_sold[i] = torch.mean(prd_sold.payoff(paths))
            price_hedge[i] = torch.mean(prd_hedge.payoff(paths))

        with open(file_path, 'wb') as file:
            pickle.dump(tuple((price_sold, price_hedge)), file, pickle.HIGHEST_PROTOCOL)

        # Initialize hedge experiment
    matHc = matHzcb = matHb = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)
    matCpl = matZcb = matB = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)

    mcSimPaths(prd_sold, mdl, rng, N_train, dTL)
    paths = mdl.x

    cpl_sold, sens_sold = fit_predict_nn_delta_vega(state0, prd_sold)

    h_c, h_zcb, pkg1, pkg2 = calc_hedge(mdl.x0, torch.tensor(0.0))

    zcb = mdl.calc_zcb(X_test, torch.tensor(0.0), (prd_sold.exerciseDate + prd_sold.delta).view(1)).flatten()
    h_b = price_sold - h_zcb * zcb - h_c * price_hedge

    r = torch.full((len(dTL), N_test), torch.nan)
    r[0, :] = mdl.calc_short_rate(X_test, torch.tensor(0.0))

    B = torch.ones((N_test,))

    V0 = price_sold - h_zcb * zcb - h_c * price_hedge - h_b * torch.tensor(1.0)
    if not torch.allclose(V0, torch.tensor(0.0)):
        print('Initial value of portfolio is not 0!')

    matHc[0, :] = h_c
    matHzcb[0, :] = h_zcb
    matHb[0, :] = h_b

    matCpl[0, :] = price_hedge
    matZcb[0, :] = zcb
    matB[0, :] = B

    # Simulate
    for k in tqdm(range(1, len(dTL))):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        # Assume calibration to know the state variables
        X_test = [x[0][k, idx_test] for x in paths]

        # Update market variables
        r[k, :] = mdl.calc_short_rate(X_test, torch.tensor(0.0))
        B *= torch.exp(0.5 * (r[k, :] + r[k - 1, :]) * dt)
        zcb = mdl.calc_zcb(X_test, torch.tensor(0.0), (prd_sold.exerciseDate + prd_sold.delta).view(1) - t).flatten()

        if t < prd_hedge.exerciseDate:
            cpl = torch.hstack(
                Parallel(n_jobs=1)(
                    delayed(cpl_mc_price)(t0=t, X=[X_test[i][j] for i in range(8)], const=const, prd=prd_hedge, dTL=dTL,
                                          N=1000)
                    for j in range(N_test)
                )
            )
            # cpl = torch.hstack([
            #    cpl_mc_price(t0=t, X=[X_test[i][j] for i in range(8)], const=const, prd=prd_hedge, dTL=dTL, N=50000).view(1)
            #    for j in range(N_test)
            # ])
        else:
            kBar = 1.0 / (1.0 + prd_hedge.delta * prd_hedge.strike)
            cpl = max0(kBar - zcb) * notional / kBar

        matCpl[k, :] = cpl
        matZcb[k, :] = zcb
        matB[k, :] = B

        # Update portfolio
        V = h_zcb * zcb + h_c * cpl + h_b * B
        if k < len(dTL) - 1:
            X_train = [x[0][k] for x in paths]
            h_c, h_zcb = calc_hedge(X_train, t)
            h_b = V - h_c * cpl - h_zcb * zcb

            matHc[k, :] = h_c
            matHzcb[k, :] = h_zcb
            matHb[k, :] = h_b

    payoff_func = lambda strike, zcb: notional / strike * max0(strike - zcb) * zcb

    zcbT = mdl.calc_zcb(X_test, torch.tensor(0.), delta).flatten()
    V_ = V * zcbT

    kBar_sold = 1 / (1 + prd_sold.delta * prd_sold.strike)

    plt.figure()
    plt.plot(zcb, V_, 'o', color='orange', label='Value of Hedge Portfolio')
    plt.plot(zcbT.sort().values, payoff_func(kBar_sold, zcbT.sort().values), color='black', label='Payoff function')
    plt.ylim(-1e4, 1e5)
    plt.xlim(0.8, 1.2)
    plt.legend()
    plt.xlabel(r'$P(T,T+\delta)$')
    plt.title('1Y6M Caplet on Zero Coupon Bond')
    plt.show()
