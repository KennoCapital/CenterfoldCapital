import torch
import itertools
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm

from application.engine.differential_NN import Neural_Approximator
from application.engine.products import EuropeanPayerSwaption
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.mcBase import mcSim, RNG, mcSimPaths
from application.utils.path_config import get_plot_path

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)


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


class EuPayerSwptAAD(torch.nn.Module):
    def __init__(self):
        super(EuPayerSwptAAD, self).__init__()

    def forward(self, X, t0, prd, N, const, simDim: int = 1, seed: int = 1234, use_av: bool = True):
        cMdl = trolleSchwartz(
            *X, *const, simDim=simDim
        )

        cPrd = EuropeanPayerSwaption(
            strike=prd.strike,
            exerciseDate=prd.exerciseDate - t0,
            delta=prd.delta,
            swapFirstFixingDate=prd.swapFirstFixingDate - t0,
            swapLastFixingDate=prd.swapLastFixingDate - t0,
            notional=prd.notional
        )

        cTL = dTL[dTL <= prd.exerciseDate - t0]
        cRng = RNG(seed=seed, use_av=use_av)
        payoffs = mcSim(cPrd, cMdl, cRng, N, cTL)

        return payoffs

    def aad(self, X, const, t0, prd, N, simDim: int = 1, seed: int = 1234, use_av: bool = True):
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = X
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
            [x, v, phi1, phi2, phi3, phi4, phi5, phi6], t0, prd, N,
            const, simDim, seed, use_av)
        payoff.backward(torch.ones((N,)), retain_graph=False)
        grad = torch.vstack([x.grad, v.grad, phi1.grad, phi2.grad, phi3.grad, phi4.grad, phi5.grad, phi6.grad]).T

        payoff = 0.5 * (payoff[:idx_half] + payoff[idx_half:])
        grad = 0.5 * (grad[:idx_half] + grad[idx_half:])

        return payoff.detach(), grad


class SwapAAD(torch.nn.Module):
    def __init__(self):
        super(SwapAAD, self).__init__()

    def forward(self, X, const, fixings, delta, K, notional, simDim: int = 1):
        cMdl = trolleSchwartz(
            *X, *const, simDim=simDim
        )

        swap = cMdl.calc_swap(X, fixings, delta, K, notional)

        return swap

    def aad(self, X, const, fixings, delta, K, notional, N, simDim: int = 1):
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = X
        x.requires_grad = True
        v.requires_grad = True
        phi1.requires_grad = True
        phi2.requires_grad = True
        phi3.requires_grad = True
        phi4.requires_grad = True
        phi5.requires_grad = True
        phi6.requires_grad = True

        swap = self.forward(X, const, fixings, delta, K, notional, simDim)
        swap.backward(torch.ones((N,)), retain_graph=False)

        v.grad = torch.zeros_like(x.grad) if v.grad is None else v.grad
        grad = torch.vstack([x.grad, v.grad, phi1.grad, phi2.grad, phi3.grad, phi4.grad, phi5.grad, phi6.grad]).T

        return swap.detach(), grad


def training_data(X, prd, t0, const, N_train, simDim=1, seed: int = 1234, use_av=True):
    cX = param_clone(*X)
    u, dudx = SwapAAD().aad(
        cX, const, prd.swapFixingDates - t0, prd.delta, prd.strike, prd.notional, N_train, simDim=simDim
    )

    cX = param_clone(*X)
    payoff, dydx = EuPayerSwptAAD().aad(
        cX, const, t0, prd, N_train, simDim=simDim, seed=seed, use_av=use_av
    )
    mask = [0, 1]
    dydu = row_wise_chain_rule(dydx[:, mask], dudx[:, mask])

    # Format training data
    x_train = torch.hstack([u.reshape(N_train, 1), X[1].reshape(-1, 1)])
    y_train = payoff.reshape(N_train, 1)
    z_train = torch.hstack([dydu, dydx[:, 1].reshape(N_train, 1)])

    return x_train, y_train, z_train, u, dudx, payoff, dydx


def diff_nn_fit_predict(x_train, y_train, z_train, x_test,
                        seed_weights, lam, hidden_units, hidden_layers, epochs, batches_per_epoch, min_batch_size
                        ):
    diff_nn = Neural_Approximator(x_train, y_train, z_train)
    diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                    hidden_layers=hidden_layers)
    diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
    y_pred, z_pred = diff_nn.predict_values_and_derivs(x_test)

    return y_pred, z_pred


if __name__ == '__main__':
    # Settings
    path_name = get_plot_path('10_ts_AAD_DiffNN_EuPayerSwpt_{}.png')
    plot_nodes = 100
    burn_in_dTL = torch.linspace(0.0, 1.0, 101)
    N_pricer = 50000

    seed = 42069
    N_train = 8192
    N_test = 256
    steps_per_year = 100
    use_av = True

    # Differential Neural Network
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = int(N_train * 0.25)
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    swapFirstFixingDate = torch.tensor(1.0)
    swapLastFixingDate = torch.tensor(5.0) + exerciseDate
    notional = torch.tensor(1e6)

    strike = torch.tensor(0.0841)

    prd = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        swapFirstFixingDate=swapFirstFixingDate,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    # Monte Carlo
    rng = RNG(seed=seed, use_av=use_av)
    dTL = torch.linspace(0.0, float(prd.exerciseDate), int(steps_per_year * prd.exerciseDate) + 1)

    # Model specification
    simDim = 1
    xt = torch.tensor([0.0])
    vt = torch.tensor([0.0194]) * 3
    phi1t = torch.tensor([0.0])
    phi2t = torch.tensor([0.0])
    phi3t = torch.tensor([0.0])
    phi4t = torch.tensor([0.0])
    phi5t = torch.tensor([0.0])
    phi6t = torch.tensor([0.0])

    kappa = torch.tensor(0.0553)
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476) * 3
    sigma = torch.tensor(0.3325)
    rho = torch.tensor(0.4615)

    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)

    varphi = torch.tensor(0.0832)

    const = kappa, theta, rho, gamma, alpha0, alpha1, sigma, varphi

    mdl = trolleSchwartz(
        xt=xt, vt=vt, phi1t=phi1t, phi2t=phi2t, phi3t=phi3t, phi4t=phi4t, phi5t=phi5t, phi6t=phi6t,
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
        simDim=simDim
    )

    # Burn in simulation and re-init
    mcSimPaths(prd, mdl, rng, N_train, burn_in_dTL)
    xt, vt, phi1t, phi2t, phi3t, phi4t, phi5t, phi6t = [x[:simDim, -1] for x in mdl.x]
    mdl = trolleSchwartz(
        xt=xt, vt=vt, phi1t=phi1t, phi2t=phi2t, phi3t=phi3t, phi4t=phi4t, phi5t=phi5t, phi6t=phi6t,
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        gamma=gamma, alpha0=alpha0, alpha1=alpha1, varphi=varphi,
        simDim=simDim
    )

    # Training data
    x_train, y_train, z_train, u, dudx, payoff, dydx = training_data(
        X=mdl.x0, prd=prd, t0=torch.tensor([0.0]), const=const, N_train=N_train, simDim=simDim, seed=seed, use_av=use_av
    )

    # Select random states as test paths
    # We use a cap based on the 90%-quantile to avoid extreme observations
    # There is no need for a lower bound as there are plenty of data there
    q = x_train.quantile(0.9, dim=0)
    cond = torch.prod(x_train <= q, dim=1).bool()
    idx_test = torch.randperm(len(x_train[cond]), generator=rng.gen)[:N_test]
    idx_test = idx_test.sort().values
    X_test = [x[0, cond][idx_test] for x in mdl.x0]

    # Test prices with Monte Carlo
    x_test = torch.hstack([
        mdl.calc_swap(X_test, prd.swapFixingDates, prd.delta, prd.strike, prd.notional).reshape(-1, 1),
        X_test[1].reshape(-1, 1)
    ])

    y_test = torch.full((N_test,), torch.nan)

    cX = [[X_test[i][j] for i in range(8)] for j in range(N_test)]
    for i in tqdm(range(N_test), desc='Calculating initial prices with MC'):
        cMdl = trolleSchwartz(*cX[i], *const, simDim=simDim)
        cRng = RNG(seed=seed, use_av=use_av)
        paths = mcSimPaths(prd, cMdl, cRng, N_pricer, dTL)
        y_test[i] = torch.mean(prd.payoff(paths))

    """
    fig, ax = plt.subplots(ncols=2)
    ax[0].hist(x_train[:, 0], bins=50, density=True, alpha=0.5, label='train', color='orange')
    ax[0].hist(x_test[:, 0], bins=50, density=True, alpha=0.5, label='test', color='black')
    ax[0].set_xlabel('Swap')
    ax[0].legend()

    ax[1].hist(x_train[:, 1], bins=50, density=True, alpha=0.5, label='train', color='orange')
    ax[1].hist(x_test[:, 1], bins=50, density=True, alpha=0.5, label='test', color='black')
    ax[1].set_xlabel('$\\nu$')
    ax[1].legend()
    plt.show()

    plt.figure()
    plt.plot(x_train[:, 0], x_train[:, 1], 'o', color='gray', alpha=0.2)
    plt.plot(x_test[:, 0], x_test[:, 1], 'o', color='black', alpha=0.5)
    plt.xlabel('Swap')
    plt.ylabel('$\\nu$')
    plt.show()
    """

    # Compare predictive performance with Monte Carlo simulations
    y_pred, z_pred = diff_nn_fit_predict(
        x_train, y_train, z_train, x_test,
        seed_weights, lam, hidden_units, hidden_layers, epochs, batches_per_epoch, min_batch_size
    )

    RMSE_value = torch.sqrt(torch.mean((y_pred.flatten() - y_test) ** 2))

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(y_test, y_test, 'r--', color='black', label='MC Price')
    ax.plot(y_test, y_pred.flatten(), 'o', color='orange', alpha=0.25, label=f'DiffReg (RMSE = {RMSE_value:.2f})')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_xlabel('MC Price')
    ax.grid(lw=0.5)
    ax.set_title('Price predictions of European Payer Swaption with Differential Neural Network')
    plt.savefig(path_name.format('price_vs_mc'), dpi=400)
    plt.show()

    # Make data used for plotting
    x_test2 = torch.tensor([
        (x, y)
        for x in torch.linspace(x_test[:, 0].quantile(0.0), x_test[:, 0].max(), plot_nodes)
        for y in torch.linspace(x_test[:, 1].quantile(0.0), x_test[:, 1].max(), plot_nodes)
    ])

    y_pred2, z_pred2 = diff_nn_fit_predict(
        x_train, y_train, z_train, x_test2,
        seed_weights, lam, hidden_units, hidden_layers, epochs, batches_per_epoch, min_batch_size
    )

    # Reshape to grids for plotting
    x1_grid = x_test2[:, 0].reshape(plot_nodes, plot_nodes)
    x2_grid = x_test2[:, 1].reshape(plot_nodes, plot_nodes)
    y_grid = y_pred2.reshape(plot_nodes, plot_nodes)
    z1_grid = z_pred2[:, 0].reshape(plot_nodes, plot_nodes)
    z2_grid = z_pred2[:, 1].reshape(plot_nodes, plot_nodes)

    test_shape = tuple([int(torch.sqrt(torch.tensor(N_test))), int(torch.sqrt(torch.tensor(N_test)))])
    x1_grid_test = x_test[:, 0].reshape(test_shape)
    x2_grid_test = x_test[:, 1].reshape(test_shape)
    y_grid_test = y_test.reshape(test_shape)

    # Price
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf_pred = ax.plot_surface(x1_grid, x2_grid, y_grid, cmap=plt.cm.magma, linewidth=0, antialiased=False, alpha=0.8)
    scatter_test = ax.scatter(x_test[:, 0], x_test[:, 1], y_test, alpha=1.0, color='black')
    ax.set_xlabel(r'$Swap$', labelpad=10)
    ax.set_ylabel(r'$\nu$')
    ax.set_zticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    cbar = fig.colorbar(surf_pred, shrink=0.5, aspect=5, format=ticker.EngFormatter())
    cbar.ax.set_title('Price')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=30, azim=-125)
    plt.savefig(path_name.format('price'), dpi=400)
    plt.show()

    # Delta
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf_pred = ax.plot_surface(x1_grid, x2_grid, z1_grid, cmap=plt.cm.magma)
    ax.set_xlabel(r'$Swap$', labelpad=10)
    ax.set_ylabel(r'$\nu$')
    ax.set_zticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    cbar = fig.colorbar(surf_pred, shrink=0.5, aspect=5)
    cbar.ax.set_title('Delta')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=30, azim=-125)
    plt.savefig(path_name.format('delta'), dpi=400)
    plt.show()

    # Vega
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf_pred = ax.plot_surface(x1_grid, x2_grid, z2_grid, cmap=plt.cm.magma)
    ax.set_xlabel(r'$Swap$', labelpad=10)
    ax.set_ylabel(r'$\nu$')
    ax.set_zticklabels([])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    cbar = fig.colorbar(surf_pred, shrink=0.5, aspect=5, format=ticker.EngFormatter())
    cbar.ax.set_title('Vega')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=30, azim=-125)
    plt.savefig(path_name.format('vega'), dpi=400)
    plt.show()
