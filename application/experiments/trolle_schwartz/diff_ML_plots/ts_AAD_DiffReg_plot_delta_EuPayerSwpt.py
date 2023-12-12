import os
import pickle
import matplotlib.pyplot as plt
import torch
from torch.func import jacfwd, vmap
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.mcBase import mcSim, RNG
from application.engine.products import EuropeanPayerSwaption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.utils.path_config import get_data_path
from functools import partial
from tqdm import tqdm

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


def calc_payoff_EuPayerSwpt(
        x, v, phi1, phi2, phi3, phi4, phi5, phi6,
        kappa, theta, rho, sigma,
        alpha0, alpha1, gamma, varphi,
        prd, rng, N, dTL, simDim
):
    cPrd = EuropeanPayerSwaption(
        strike=prd.strike,
        exerciseDate=prd.exerciseDate,
        delta=prd.delta,
        swapFirstFixingDate=prd.swapFirstFixingDate,
        swapLastFixingDate=prd.swapLastFixingDate,
        notional=prd.notional
    )
    cRng = RNG(seed=rng.seed, use_av=rng.use_av)
    cMdl = trolleSchwartz(
        xt=x,
        vt=v,
        phi1t=phi1,
        phi2t=phi2,
        phi3t=phi3,
        phi4t=phi4,
        phi5t=phi5,
        phi6t=phi6,
        kappa=kappa,
        theta=theta,
        rho=rho,
        sigma=sigma,
        alpha0=alpha0,
        alpha1=alpha1,
        gamma=gamma,
        varphi=varphi,
        simDim=simDim
    )
    cTL = dTL
    payoff = mcSim(cPrd, cMdl, cRng, N, cTL).reshape(N, -1)
    return payoff, payoff


def calc_swap(
    x, v, phi1, phi2, phi3, phi4, phi5, phi6,
    kappa, theta, rho, sigma, alpha0, alpha1, gamma, varphi,
    fixings, delta, K, N,
    simDim: int = 1
):
    #
    X = (x, v, phi1, phi2, phi3, phi4, phi5, phi6)
    cMdl = trolleSchwartz(
        xt=x,
        vt=v,
        phi1t=phi1,
        phi2t=phi2,
        phi3t=phi3,
        phi4t=phi4,
        phi5t=phi5,
        phi6t=phi6,
        kappa=kappa,
        theta=theta,
        rho=rho,
        sigma=sigma,
        alpha0=alpha0,
        alpha1=alpha1,
        gamma=gamma,
        varphi=varphi,
        simDim=simDim
    )

    swap = cMdl.calc_swap(X=X, t=fixings, delta=delta, K=K, N=N).flatten().reshape(-1, 1)
    return swap, swap


if __name__ == '__main__':
    filename_path = get_data_path('ts_EuPayerSwpt_testset.pkl')

    seed = 1234
    N_train = 512
    N_test = 32  # Total test cases are N_test squared
    steps_per_year = 100
    use_av = True

    # Monte Carlo specification
    rng = RNG(seed=seed, use_av=use_av)

    # Setup Differential Regressor, and Scalar
    deg = 5
    alpha = 1.0
    include_interactions = True
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=include_interactions)
    scalar = DifferentialStandardScaler()

    # Model specification
    simDim = 1
    xt = (-0.5 - 0.5) * torch.rand((N_train, ), generator=rng.gen) + 0.5
    vt = (1e-6 - 0.5) * torch.rand((N_train, ), generator=rng.gen) + 0.5
    phi1t = torch.tensor([0.0]) * torch.ones(N_train)
    phi2t = torch.tensor([0.075]) * torch.ones(N_train)
    phi3t = torch.tensor([0.040]) * torch.ones(N_train)
    phi4t = torch.tensor([0.200]) * torch.ones(N_train)
    phi5t = torch.tensor([0.060]) * torch.ones(N_train)
    phi6t = torch.tensor([0.165]) * torch.ones(N_train)

    kappa = torch.tensor(0.0553) 
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476)
    sigma = torch.tensor(0.3325)
    rho = torch.tensor(0.4615)

    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)

    varphi = torch.tensor(0.0832)

    const = kappa, theta, sigma, rho, gamma, alpha0, alpha1, varphi

    mdl = trolleSchwartz(
        xt=xt, vt=vt, phi1t=phi1t, phi2t=phi2t, phi3t=phi3t, phi4t=phi4t, phi5t=phi5t, phi6t=phi6t,
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
        simDim=simDim
    )

    # Product specification
    exerciseDate = torch.tensor(1.0)
    strike = torch.tensor(0.08)
    delta = torch.tensor(0.25)
    swapFirstFixingDate = torch.tensor(1.0)
    swapLastFixingDate = torch.tensor(5.0) + exerciseDate
    notional = torch.tensor(1e6)

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    dTL = torch.linspace(0.0, float(exerciseDate), int(exerciseDate * steps_per_year) + 1)

    # Auxiliary functions
    f_V = partial(
        calc_payoff_EuPayerSwpt,
        kappa=kappa, theta=theta, rho=rho, sigma=sigma, alpha0=alpha0, alpha1=alpha1, gamma=gamma, varphi=varphi,
        prd=prd, rng=rng, N=N_train, dTL=dTL, simDim=1
    )
    f_swap = partial(
        calc_swap,
        # x, v, phi1, phi2, phi3, phi4, phi5, phi6,
        kappa=kappa, theta=theta, rho=rho, sigma=sigma, alpha0=alpha0, alpha1=alpha1, gamma=gamma,
        varphi=varphi,
        fixings=prd.swapFixingDates, delta=prd.delta, K=prd.strike, N=prd.notional, simDim=1
    )

    # Compute Jacobians and values
    ones = torch.ones(N_train, 1)
    state = mdl.x0
    Jswap, swap = jacfwd(f_swap, argnums=(0, 1, 2, 3, 4, 5, 6, 7), has_aux=True)(*state)
    Jpayoff, payoff = jacfwd(f_V, argnums=(0, 1, 2, 3, 4, 5, 6, 7), has_aux=True, randomness='same')(*state)

    dU_dParam = torch.hstack([J.squeeze() @ ones for J in Jswap])
    dPayoff_dParam = torch.hstack([J.squeeze() @ ones for J in Jpayoff])

    # Trick using the chain-rule as stated in "Fixed Income Derivatives" p. 28
    dVdu = torch.full((N_train, 1), torch.nan)
    for i in range(N_train):
        dVdu[i, 0] = torch.pinverse(dU_dParam[i, :].reshape(1, -1).T) @ dPayoff_dParam[i, :].reshape(-1, 1)

    # Make test dataset for comparison
    xt_test = torch.linspace(xt.min(), xt.max(), N_test)
    vt_test = torch.linspace(vt.min(), vt.max(), N_test)

    if os.path.isfile(filename_path):
        with open(filename_path, 'rb') as file:
            x_test, y_test, z_test, swap_test = pickle.load(file)
    else:
        y_test = torch.full((N_test, N_test), torch.nan)
        swap_test = torch.full((N_test, N_test), torch.nan)
        for i, xt_ in tqdm(enumerate(xt_test), total=N_test, desc='Calculating test dataset with MC'):
            for j, vt_ in enumerate(vt_test):
                cMdl = trolleSchwartz(
                    xt=xt_, vt=vt_,
                    phi1t=phi1t[0], phi2t=phi2t[0], phi3t=phi3t[0], phi4t=phi4t[0], phi5t=phi5t[0], phi6t=phi6t[0],
                    kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                    gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
                    simDim=simDim
                )
                swap_test[i, j] = cMdl.calc_swap(cMdl.x0, prd.swapFixingDates, prd.delta, prd.strike, prd.notional)
                y_test[i, j] = torch.mean(mcSim(prd, cMdl, rng, 50000, dTL))

        x_test = torch.cartesian_prod(swap_test[:, 0], vt_test)
        delta_test = y_test.diff(dim=0) / swap_test.diff(dim=0)
        vega_test = y_test.diff(dim=1) / vt_test.diff()
        z_test = [delta_test, vega_test]

        with open(filename_path, 'wb') as file:
            test_data = tuple((x_test, y_test, z_test, swap_test))
            pickle.dump(test_data, file, pickle.HIGHEST_PROTOCOL)

    # Make training dataset
    x_train = torch.hstack([swap, vt.reshape(-1, 1)])
    y_train = payoff
    z_train = torch.hstack([dVdu, dPayoff_dParam[:, 1].reshape(-1, 1)])

    # Differential regression
    x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)

    diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)

    x_test_scaled, _, _ = scalar.transform(x_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(x_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    ''' Plot Results '''

    # Data for plotting a surface
    x_ = x_test[:, 0].reshape(N_test, N_test)
    y_ = x_test[:, 1].reshape(N_test, N_test)
    z_price = y_pred.reshape(N_test, N_test)
    z_delta = z_pred[:, 0].reshape(N_test, N_test)
    z_vega = z_pred[:, 1].reshape(N_test, N_test)
    zero = torch.zeros(N_test, N_test)

    vt_test = torch.hstack([vt for i in range(N_test)])

    # Price
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    surf_train = ax.scatter(swap, vt, payoff, c='gray', alpha=1.0)
    surf_test = ax.scatter(x_test[:, 0], x_test[:, 1], y_test.flatten(), c='black')
    surf_pred = ax.plot_surface(x_, y_, z_price, cmap=plt.cm.magma)

    ax.set_xlabel('Swap(0)')
    ax.set_ylabel('v(0)')
    ax.set_zlabel('Payoff')

    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()

    # Delta
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    surf_train = ax.scatter(swap, vt, z_train[:, 0], c='gray', alpha=1.0)
    surf_test = ax.scatter(x_test[:, 0].reshape(N_test, N_test)[1:].flatten(),
                           x_test[:, 1].reshape(N_test, N_test)[1:].flatten(),
                           z_test[0].flatten(), c='black')
    surf_pred = ax.plot_surface(x_, y_, z_delta, cmap=plt.cm.magma)

    ax.set_xlabel('Swap(0)')
    ax.set_ylabel('v(0)')
    ax.set_zlabel('Delta')

    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()

    # Vega
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, sharex='all', sharey='all')
    #surf_train = ax.scatter(swap, vt, z_train[:, 1] / 10000, c='gray', alpha=1.0)
    surf_test = ax.scatter(x_test[:, 0].reshape(N_test, N_test)[1:].flatten(),
                           x_test[:, 1].reshape(N_test, N_test)[1:].flatten(),
                           z_test[1].flatten() / 10000, c='black')
    surf_pred = ax.plot_surface(x_, y_, z_vega / 10000, cmap=plt.cm.magma)

    ax.set_xlabel('Swap(0)')
    ax.set_ylabel('v(0)')
    ax.set_zlabel('Vega (per bp)')
    ax.set_zlim(-50, 150)

    fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    plt.show()

