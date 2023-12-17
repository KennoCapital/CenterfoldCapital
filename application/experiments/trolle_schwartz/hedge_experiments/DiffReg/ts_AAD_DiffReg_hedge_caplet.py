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

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

MAX_PROCESSES = os.cpu_count() - 1
print(f'Settings MAX_PROCESSES = {MAX_PROCESSES}')


def calc_hedge_coef(X_train, t0,
                    prd_sold, prd_hedge, N_train,
                    const, diff_reg, scalar,
                    simDim: int = 1, seed: int = 1234, use_av: bool = True):
    _, z_sold, _, _, _ = estimate_greeks(
        X=X_train, t0=t0, prd=prd_sold, N_train=N_train,
        const=const, diff_reg=diff_reg, scalar=scalar, simDim=simDim,
        seed=seed, use_av=use_av
    )
    _, z_hedge, _, _, _ = estimate_greeks(
        X=X_train, t0=t0, prd=prd_hedge, N_train=N_train,
        const=const, diff_reg=diff_reg, scalar=scalar, simDim=simDim,
        seed=seed, use_av=use_av
    )

    h_c = z_sold[:, 1] / z_hedge[:, 1]
    h_zcb = z_sold[:, 0] - z_hedge[:, 0] * h_c

    return h_c, h_zcb


def cpl_mc_price(t0, X, const, prd, dTL, N: int = 50000, simDim: int = 1, seed: int = 1234, use_av: bool = True):
    cPrd = CapletAsPutOnZCB(
        strike=prd.strike,
        exerciseDate=prd.exerciseDate - t0,
        delta=delta,
        notional=prd.notional
    )
    cMdl = trolleSchwartz(*X, *const, simDim)
    cRng = RNG(seed=seed, use_av=use_av)
    cTL = dTL[dTL <= prd.exerciseDate - t0]
    payoff = mcSim(cPrd, cMdl, cRng, N, cTL)
    return torch.mean(payoff).view(1)


def estimate_greeks(X, t0, prd, N_train, const,
                    diff_reg=None, scalar=None,
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

    if diff_reg is None or scalar is None:
        return x_train, y_train, z_train

    # Make predictions
    f_pred = partial(diff_reg_fit_predict, diff_reg=diff_reg, scalar=scalar)
    y_pred, z_pred = f_pred(x_train, y_train, z_train, x_train)
    return y_pred, z_pred, x_train, y_train, z_train


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
        payoff.backward(torch.ones((N, )), retain_graph=False)
        grad = torch.vstack([x.grad, v.grad, phi1.grad, phi2.grad, phi3.grad, phi4.grad, phi5.grad, phi6.grad]).T

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
        zcb.backward(torch.ones((simDim, len(tenor), N, )), retain_graph=False)

        v.grad = torch.zeros_like(x.grad) if v.grad is None else v.grad
        grad = torch.vstack([x.grad, v.grad, phi1.grad, phi2.grad, phi3.grad, phi4.grad, phi5.grad, phi6.grad]).T

        return zcb.detach(), grad


if __name__ == '__main__':
    # Settings
    file_path = get_data_path('ts_cpl_mc_prices.pkl')
    burn_in_dTL = torch.linspace(0.0, 0.5, 51)
    N_mc_price = 50000

    seed = 1234
    N_train = 8196
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
    strike_hedge = torch.tensor(0.09)

    prd_sold = CapletAsPutOnZCB(
        strike=strike,
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
    xt = torch.tensor([0.0])
    vt = torch.tensor([0.0194])
    phi1t = torch.tensor([0.0])
    phi2t = torch.tensor([0.0])
    phi3t = torch.tensor([0.0])
    phi4t = torch.tensor([0.0])
    phi5t = torch.tensor([0.0])
    phi6t = torch.tensor([0.0])

    kappa = torch.tensor(0.0553)
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476)
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

    # Burn in simulation and re-init
    mcSimPaths(prd_sold, mdl, rng, N_train, burn_in_dTL)
    state = [x[:simDim, -1] for x in mdl.x]

    mdl = trolleSchwartz(
        *state,
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
        simDim=simDim
    )

    """
    T = torch.linspace(0.0, 30.0, 121)
    fwd = mdl.calc_instant_fwd(mdl.x0, torch.tensor(0.0), T)

    def f_to_zcb(fwd, T):
        return torch.exp(- torch.cumsum(0.5 * (fwd[:-1] + fwd[1:]) * T.diff().reshape(-1, 1), dim=0))

     plt.plot(T[1:], f_to_zcb(fwd, T)[:, :100], color='black', alpha=0.2)
     plt.xlabel('$T$')
     plt.ylabel('$P(0,T)$')
     plt.show()

     for i in range(100):
        plt.plot(torch.linspace(0.0, 30.0, 121), fwd[:, i], color='black', alpha=0.2)
     plt.ylabel('$f(t,T)$')
     plt.xlabel('$T$')
     plt.show()
    """

    y_pred, z_pred, x_train, y_train, z_train = estimate_greeks(mdl.x0, torch.tensor(0.0), prd_sold, N_train, const, diff_reg, scalar)

    fig, ax = plt.subplots(2, sharex='all')
    ax[0].plot(x_train[:, 0], y_train, 'o', color='gray', alpha=0.5/8)
    ax[0].plot(x_train[:, 0], y_pred, 'o', color='orange', alpha=0.25)

    ax[1].plot(x_train[:, 0], z_train[:, 0], 'o', color='gray', alpha=0.5/8)
    ax[1].plot(x_train[:, 0], z_pred[:, 0], 'o', color='orange', alpha=0.25)

    ax[1].set_xlabel('ZCB')
    ax[0].set_ylabel('Cpl')
    ax[1].set_ylabel('Delta')
    plt.show()

    plt.plot(x_train[:, 0], z_train[:, 1], 'o', color='gray', alpha=0.5/4)
    plt.plot(x_train[:, 0], z_pred[:, 1], 'o', color='orange', alpha=0.25)
    plt.ylim(z_pred[:, 1].min() * 1.05, z_pred[:, 1].max() * 1.05)
    plt.show()

    # Select random states as test paths
    idx_test = torch.randperm(N_train, generator=rng.gen)[:N_test]
    X_test = [x[0, idx_test] for x in mdl.x0]
    xt_test, vt_test, phi1t_test, phi2t_test, phi3t_test, phi4t_test, phi5t_test, phi6t_test = X_test

    if os.path.exists(get_data_path('ts_cpl_mc_full.pkl')):
        with open(get_data_path('ts_cpl_mc_full.pkl'), 'rb') as file:
            zcb_sold, v_sold, price_sold, delta_sold, vega_sold, zcb_hedge, v_hedge, price_hedge, delta_hedge, vega_hedge = pickle.load(file)
    else:
        price_sold = torch.full((N_test, ), torch.nan)
        price_hedge = torch.full((N_test, ), torch.nan)

        zcb_sold = torch.full((N_test, ), torch.nan)
        zcb_hedge = torch.full((N_test, ), torch.nan)
        v_sold = torch.full((N_test, ), torch.nan)
        v_hedge = torch.full((N_test, ), torch.nan)

        delta_sold = torch.full((N_test, ), torch.nan)
        delta_hedge = torch.full((N_test, ), torch.nan)
        vega_sold = torch.full((N_test, ), torch.nan)
        vega_hedge = torch.full((N_test, ), torch.nan)

        cX = [[X_test[i][j] * torch.ones((N_mc_price,)) for i in range(8)] for j in range(N_test)]
        for i in tqdm(range(N_test), desc='Calculating initial prices with MC'):

            cX = [xt_test[i], vt_test[i], phi1t_test[i], phi2t_test[i], phi3t_test[i], phi4t_test[i], phi5t_test[i], phi6t_test[i]]
            
            cMdl = trolleSchwartz(*cX, *const, simDim)
            cRng = RNG(seed=seed, use_av=use_av)
            paths = mcSimPaths(prd_sold, cMdl, cRng, N_mc_price, dTL)
            price_sold[i] = torch.mean(prd_sold.payoff(paths))
            price_hedge[i] = torch.mean(prd_hedge.payoff(paths))
            """
            x_, y_, z_ = estimate_greeks(X=cX[i], t0=torch.tensor(0.0), prd=prd_sold, N_train=10000, const=const)
            zcb_sold[i] = torch.mean(x_[:, 0])
            v_sold[i] = torch.mean(x_[:, 1])
            price_sold[i] = torch.mean(y_)
            delta_sold[i] = torch.mean(z_[:, 0])
            vega_sold[i] = torch.mean(z_[:, 1])

            x_, y_, z_ = estimate_greeks(X=cX[i], t0=torch.tensor(0.0), prd=prd_hedge, N_train=10000, const=const)
            zcb_hedge[i] = torch.mean(x_[:, 0])
            v_hedge[i] = torch.mean(x_[:, 1])
            price_hedge[i] = torch.mean(y_)
            delta_hedge[i] = torch.mean(z_[:, 0])
            vega_hedge[i] = torch.mean(z_[:, 1])
            """
        # filepath
        with open(get_data_path('ts_cpl_mc_full.pkl'), 'wb') as file:
            test_data = zcb_sold, v_sold, price_sold, delta_sold, vega_sold, zcb_hedge, v_hedge, price_hedge, delta_hedge, vega_hedge
            pickle.dump(tuple(test_data), file, pickle.HIGHEST_PROTOCOL)

    # Auxiliary function
    calc_hedge = partial(calc_hedge_coef,
                         prd_sold=prd_sold, prd_hedge=prd_hedge, N_train=N_train, const=const,
                         diff_reg=diff_reg, scalar=scalar, simDim=simDim, seed=seed, use_av=use_av)

    # Compare initial fit
    out_sold = estimate_greeks(X=mdl.x0, t0=torch.tensor(0.0), prd=prd_sold, N_train=N_train,
                               const=const, diff_reg=diff_reg, scalar=scalar)
    out_hedge = estimate_greeks(X=mdl.x0, t0=torch.tensor(0.0), prd=prd_hedge, N_train=N_train,
                                const=const, diff_reg=diff_reg, scalar=scalar)
    y_pred_sold, z_pred_sold, x_train_sold, y_train_sold, z_train_sold = out_sold
    y_pred_hedge, z_pred_hedge, x_train_hedge, y_train_hedge, z_train_hedge = out_hedge




    # Initialize hedge experiment
    mcSimPaths(prd_sold, mdl, rng, N_train, dTL)
    paths = mdl.x

    matHc = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)
    matHzcb = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)
    matHb = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)
    matCpl = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)
    matZcb = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)
    matB = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)
    matV = torch.full(size=(len(dTL), N_test), fill_value=torch.nan)

    h_c, h_zcb = calc_hedge(mdl.x0, torch.tensor(0.0))[idx_test, :]

    zcb = mdl.calc_zcb(X_test, torch.tensor(0.0), (prd_sold.exerciseDate + prd_sold.delta).view(1)).flatten()
    h_b = price_sold - h_zcb * zcb - h_c * price_hedge

    r = torch.full((len(dTL), N_test), torch.nan)
    r[0, :] = mdl.calc_short_rate(X_test, torch.tensor(0.0))

    B = torch.ones((N_test, ))

    V0 = price_sold - h_zcb * zcb - h_c * price_hedge - h_b * torch.tensor(1.0)
    if not torch.allclose(V0, torch.tensor(0.0)):
        print('Initial value of portfolio is not 0!')

    matHc[0, :] = h_c
    matHzcb[0, :] = h_zcb
    matHb[0, :] = h_b

    matV[0, :] = V0

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
                Parallel(n_jobs=MAX_PROCESSES)(
                    delayed(cpl_mc_price)(t0=t, X=[X_test[i][j] for i in range(8)], const=const, prd=prd_hedge, dTL=dTL[0::2], N=10000)
                    for j in range(N_test)
                )
            )
            #cpl = torch.hstack([
            #    cpl_mc_price(t0=t, X=[X_test[i][j] for i in range(8)], const=const, prd=prd_hedge, dTL=dTL, N=50000).view(1)
            #    for j in range(N_test)
            #])
        else:
            kBar = 1.0 / (1.0 + prd_hedge.delta * prd_hedge.strike)
            cpl = max0(kBar - zcb)

        matCpl[k, :] = cpl
        matZcb[k, :] = zcb
        matB[k, :] = B

        # Update portfolio
        V = h_zcb * zcb + h_c * cpl + h_b * B
        matV[k, :] = V

        if k < len(dTL) - 1:
            X_train = [x[0][k] for x in paths]
            h_c, h_zcb = calc_hedge(X_train, t)[idx_test, :]
            h_b = V - h_c * cpl - h_zcb * zcb

            matHc[k, :] = h_c
            matHzcb[k, :] = h_zcb
            matHb[k, :] = h_b

    # Export
    #with open(get_data_path('ts_delta_hedge.pkl'), 'wb') as file:
    #    pickle.dump(tuple((matCpl, matZcb, matB, matHc, matHzcb, matHb, matV)), file, pickle.HIGHEST_PROTOCOL)

    # Import
    #with open(get_data_path('ts_delta_hedge.pkl'), 'rb') as file:
    #    matCpl, matZcb, matB, matHc, matHzcb, matHb, matV = pickle.load(file)

    #plt.plot(matZcb[-1], matV[-1], 'o')
    #plt.show()