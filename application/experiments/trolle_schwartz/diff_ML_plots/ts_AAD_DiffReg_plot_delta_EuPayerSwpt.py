import matplotlib.pyplot as plt

import torch
from torch.func import jacfwd
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.mcBase import mcSim, RNG
from application.engine.products import EuropeanPayerSwaption
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from functools import partial

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


def calc_payoff_EuPayerSwpt(
        v0, varphi, gamma, kappa, theta, rho, sigma, alpha0, alpha1,
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
        v0=v0,
        gamma=gamma,
        kappa=kappa,
        theta=theta,
        rho=rho,
        sigma=sigma,
        alpha0=alpha0,
        alpha1=alpha1,
        varphi=varphi,
        simDim=simDim
    )
    cTL = dTL
    payoff = mcSim(cPrd, cMdl, cRng, N, cTL).reshape(N, -1)
    return payoff, payoff


def calc_swap(
    v0, varphi, gamma, kappa, theta, rho, sigma, alpha0, alpha1,
    x, v, phi1, phi2, phi3, phi4, phi5, phi6,
    fixings, delta, K, N,
    simDim: int = 1
):
    #
    X = (x, v, phi1, phi2, phi3, phi4, phi5, phi6)
    cMdl = trolleSchwartz(
        v0=v0,
        gamma=gamma,
        kappa=kappa,
        theta=theta,
        rho=rho,
        sigma=sigma,
        alpha0=alpha0,
        alpha1=alpha1,
        varphi=varphi,
        simDim=simDim
    )

    swap = cMdl.calc_swap(X=X, t=fixings, delta=delta, K=K, N=N).flatten().reshape(-1, 1)
    return swap, swap


if __name__ == '__main__':
    seed = 1234
    N_train = 512
    N_test = 16 ** 2
    steps_per_year = 100
    use_av = True

    # Model specification
    kappa = torch.tensor(0.0553) 
    sigma = torch.tensor(0.3325) 
    alpha0 = torch.tensor(0.045) 
    alpha1 = torch.tensor(0.131) 
    gamma = torch.tensor(0.3341) 
    rho = torch.tensor(0.4615) 
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476) 
    #v0 = theta
    #varphi = torch.tensor(0.0832)

    #v0 = torch.ones(N_train) * theta
    #v0 = torch.linspace(1E-6, 0.5, N_train)
    #varphi = torch.linspace(-0.02, 0.15, N_train)
    #grid = torch.stack(torch.meshgrid(v0, varphi)).mT.reshape(-1, 2)

    # Random grid in U[a, b] x U[c, d]
    v0 = (1E-6 - 0.5) * torch.rand(N_train) + 0.5
    varphi = (-0.02 - 0.15) * torch.rand(N_train) + 0.15
    #grid = torch.hstack((v0, varphi))

    param = v0, varphi, gamma, kappa, theta, rho, sigma, alpha0, alpha1

    mdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)

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

    # Monte Carlo specification
    rng = RNG(seed=seed, use_av=use_av)
    dTL = torch.linspace(0.0, float(exerciseDate), int(exerciseDate * steps_per_year) + 1)

    # Auxiliary functions
    x, v, phi1, phi2, phi3, phi4, phi5, phi6 = mdl.x0
    f_V = partial(calc_payoff_EuPayerSwpt, prd=prd, rng=rng, N=N_train, dTL=dTL,
                  #gamma=gamma, kappa=kappa, theta=theta, rho=rho, sigma=sigma, alpha0=alpha0, alpha1=alpha1
                  simDim=1)
    f_swap = partial(calc_swap, fixings=prd.swapFixingDates, delta=prd.delta, K=prd.strike, N=prd.notional,
                     #gamma=gamma, kappa=kappa, theta=theta, rho=rho, sigma=sigma, alpha0=alpha0, alpha1=alpha1,
                     #x=x, v=v, phi1=phi1, phi2=phi2, phi3=phi3, phi4=phi4, phi5=phi5, phi6=phi6,
                     simDim=1)

    # Compute Jacobians and values (argnums=(0, 1) allow for changing v0 and varphi)
    ones = torch.ones(N_train)
    state = mdl.x0
    Jswap, swap = jacfwd(f_swap, argnums=(0, 1), has_aux=True)(*param, *state)
    Jpayoff, payoff = jacfwd(f_V, argnums=(0, 1), has_aux=True, randomness='same')(*param)
    dU_dParam = torch.hstack([J @ ones for J in Jswap])
    dPayoff_dParam = torch.hstack([J @ ones for J in Jpayoff])

    # Trick using the chain-rule as stated in "Fixed Income Derivatives" p. 28
    dVdu = torch.full((N_train, 1), torch.nan)
    for i in range(N_train):
        dVdu[i, 0] = torch.pinverse(dU_dParam[i, :].reshape(1, -1).T) @ dPayoff_dParam[i, :].reshape(-1, 1)

    # Setup Differential Regressor, and Scalar
    deg = 5
    alpha = 1.0
    include_interactions = True
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=include_interactions)
    scalar = DifferentialStandardScaler()

    x_train = torch.hstack([swap, v0.reshape(-1, 1)])
    y_train = payoff
    z_train = torch.hstack([dVdu, dPayoff_dParam[:, 1].reshape(-1, 1)])

    swap_test = torch.linspace(swap.min(), swap.max(), 32)
    v0_test = torch.linspace(v0.min(), v0.max(), 32)
    x_test = torch.cartesian_prod(swap_test, v0_test)

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
    surf_train = ax.scatter(swap, v0, payoff, c='gray', alpha=1.0)
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


