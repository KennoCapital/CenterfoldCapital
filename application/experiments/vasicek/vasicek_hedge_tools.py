import numpy as np
import torch
import itertools

from matplotlib import pyplot as plt
from scipy.stats import linregress
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.differential_NN import Neural_Approximator
from application.utils.path_config import get_plot_path


def training_data(r0_vec: torch.Tensor, t0: float, calc_dU_dr, calc_dPrd_dr, use_av: bool = True):
    """
    param r0_vec:           1D vector of short rates to generate training data from
    param t0:               Current market time, effects time to expiry and fixings in the training
    param calc_dU_dr:       Function for calculating the derivative of the underlying wrt. to r
    param calc_dPrd_dr:     Function for calculating the derivative of the product wrt. to r
    param use_av:           Use antithetic variates to reduce variance of both y- and z-labels

    Note: z-labels are calculated using the (matrix) equations on
     page 28 in Fixed Income Derivates - Lecture Notes (2013) by Linderstrøm

    returns:                tuple of (x_train, y_train, z_train)
    """
    idx_half = len(r0_vec)
    if use_av:
        r0_vec = torch.hstack([r0_vec, r0_vec])

    x_train, dxdr = calc_dU_dr(r0_vec, t0)
    y_train, dydr = calc_dPrd_dr(r0_vec, t0)

    # Ensure that 1d cases are formatted as column vectors
    if x_train.dim() == 1:
        x_train = x_train.reshape(-1, 1)
    if dxdr.dim() == 1:
        dxdr = dxdr.reshape(-1, 1)
    if y_train.dim() == 1:
        y_train = y_train.reshape(-1, 1)
    if dydr.dim() == 1:
        dydr = dydr.reshape(-1, 1)

    num_features = x_train.shape[1]

    if num_features > 1:
        # General (multi-dimensional) case
        solve_rowwise = lambda dxdr_, dydr_: (torch.pinverse(dxdr_.T) @ dydr_.T).reshape(-1, num_features).sum(dim=0)
        equations = (
            (dxdr[i, :].reshape(-1, 1), dydr[i, :].reshape(-1, 1)) for i in range(len(r0_vec))
        )
        solutions = itertools.starmap(solve_rowwise, equations)
        z_train = torch.vstack(list(solutions))
    else:
        # 1d case can be computed all at once
        z_train = dydr / dxdr

    if use_av:
        x_train = x_train[:idx_half]
        y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
        z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])
    return x_train, y_train, z_train


def diff_reg_fit_predict(u_vec: torch.Tensor,
                         r0_vec: torch.Tensor,
                         t0: float,
                         calc_dU_dr,
                         calc_dPrd_dr,
                         diff_reg: DifferentialPolynomialRegressor,
                         use_av: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """
    param u_vec:            Underlying market variables
    param r0_vec:           1D vector of short rates to generate training data from
    param t0:               Current market time, effects time to expiry and fixings in the training
    param calc_dU_dr:       Function for calculating the derivative of the underlying wrt. to r
    param calc_dPrd_dr:     Function for calculating the derivative of the product wrt. to r
    param use_av:           Use antithetic variates to reduce variance of both y- and z-labels

    returns:                tuple of (y_pred, z_pred)
    """
    scalar = DifferentialStandardScaler()
    if u_vec.dim() == 1:
        X_test = u_vec.reshape(-1, 1)
    elif u_vec.dim() == 2:
        X_test = u_vec.reshape(-1, u_vec.shape[1])
    else:
        raise ValueError

    x_train, y_train, z_train = training_data(
        r0_vec=r0_vec, t0=t0, calc_dU_dr=calc_dU_dr, calc_dPrd_dr=calc_dPrd_dr, use_av=use_av
    )

    x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)

    diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)

    X_test_scaled, _, _ = scalar.transform(X_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    return y_pred, z_pred


def calc_delta_diff_nn(u_vec: torch.Tensor,
                        r0_vec: torch.Tensor,
                        t0: float,
                        calc_dU_dr,
                        calc_dPrd_dr,
                        nn_Params: dict,
                        use_av: bool) -> torch.Tensor:
    """
    param u_vec:            1D vector of the underlying market variable
    param r0_vec:           1D vector of short rates to generate training data from
    param t0:               Current market time, effects time to expiry and fixings in the training
    param calc_dU_dr:       Function for calculating the derivative of the underlying wrt. to r
    param calc_dPrd_dr:     Function for calculating the derivative of the product wrt. to r
    param use_av:           Use antithetic variates to reduce variance of both y- and z-labels

    returns:                1D vector of predicted deltas for `u_vec`
    """

    N_train = nn_Params['N_train']
    seed_weights = nn_Params['seed_weights']
    lam = nn_Params['lam']
    hidden_units = nn_Params['hidden_units']
    hidden_layers = nn_Params['hidden_layers']
    epochs = nn_Params['epochs']
    batches_per_epoch = nn_Params['batches_per_epoch']
    min_batch_size = nn_Params['min_batch_size']

    if any(v is None for v in nn_Params.values()):
        raise ValueError(f'Missing parameters to set NN')

    X_test = u_vec.reshape(-1, 1)

    X_train, y_train, z_train = training_data(r0_vec=r0_vec, t0=t0,
                                                 calc_dU_dr=calc_dU_dr,
                                                 calc_dPrd_dr=calc_dPrd_dr,
                                                 use_av=use_av
                                                 )

    # Setup Differential Neutral Network
    diff_nn = Neural_Approximator(X_train, y_train, z_train)
    diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                    hidden_layers=hidden_layers)
    diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)

    _, z_pred = diff_nn.predict_values_and_derivs(X_test)

    return z_pred.flatten()


def log_plotter(X, Y, title_add : str, save: bool, file_name: str = None):
    # add convergence order line
    x = np.log(X)
    y = np.log(Y)
    res = linregress(x, y)
    fit_y_log = res.slope * x + res.intercept

    plt.figure()
    plt.suptitle(title_add)
    plt.title(f'convergence order = {res.slope:.2f}')
    plt.plot(x, fit_y_log, '--', color='red')
    plt.plot(x, y, 'o-', color='blue')

    plt.xlabel('steps per fixing')
    plt.ylabel('std. dev. of hedge error')

    plt.xticks(ticks=x, labels=X)
    plt.yticks(ticks=y, labels=np.round(y, 2))

    if save:
        plt.savefig(get_plot_path(f'{file_name}.png'), dpi=400)
    plt.show()

def log_plotter_without_conv(X, Y, title_add : str, xLabel: str, yLabel: str, save: bool, file_name: str = None):
    if not (xLabel and yLabel):
        raise ValueError("Need to specify X and Y axis when plotting!")
    x = np.log(X)
    y = np.log(Y)

    plt.figure()
    plt.suptitle(title_add)
    plt.plot(x, y, 'o-', color='orange')

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    plt.xticks(ticks=x, labels=X)

    if save:
        plt.savefig(get_plot_path(f'{file_name}.png'), dpi=400)
    plt.show()
