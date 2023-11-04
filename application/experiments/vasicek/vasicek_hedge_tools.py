import torch
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor


def training_data(r0_vec: torch.Tensor, t0: float, calc_dU_dr, calc_dPrd_dr, use_av: bool = True):
    """
    param r0_vec:           1D vector of short rates to generate training data from
    param t0:               Current market time, effects time to expiry and fixings in the training
    param calc_dU_dr:       Function for calculating the derivative of the underlying wrt. to r
    param calc_dPrd_dr:     Function for calculating the derivative of the product wrt. to r
    param use_av:           Use antithetic variates to reduce variance of both y- and z-labels

    returns:                tuple of (x_train, y_train, z_train)
    """
    N_train = len(r0_vec)
    if use_av:
        # x_train[i] = x_train[i + N_train],  for all i, when using AV
        r0_vec = torch.concat([r0_vec, r0_vec])

    u, dUdr = calc_dU_dr(r0_vec, t0)
    y, dydr = calc_dPrd_dr(r0_vec, t0)

    x_train = u.reshape(-1, u.shape[1])
    y_train = y.reshape(-1, 1)
    z_train = dydr.reshape(-1, 1) / dUdr.reshape(-1, dUdr.shape[1])

    if use_av:
        idx_half = N_train
        x_train = x_train[:idx_half]
        y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
        z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])

    return x_train, y_train, z_train


def calc_delta_diff_reg(u_vec: torch.Tensor,
                        r0_vec: torch.Tensor,
                        t0: float,
                        calc_dU_dr,
                        calc_dPrd_dr,
                        diff_reg: DifferentialPolynomialRegressor,
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
    scalar = DifferentialStandardScaler()
    X_test = u_vec.reshape(-1, 1)

    x_train, y_train, z_train = training_data(r0_vec=r0_vec,
                                              t0=t0,
                                              calc_dU_dr=calc_dU_dr,
                                              calc_dPrd_dr=calc_dPrd_dr,
                                              use_av=use_av)

    x_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(x_train, y_train, z_train)

    diff_reg.fit(x_train_scaled, y_train_scaled, z_train_scaled)

    X_test_scaled, _, _ = scalar.transform(X_test, None, None)
    y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

    _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

    return z_pred.flatten()
