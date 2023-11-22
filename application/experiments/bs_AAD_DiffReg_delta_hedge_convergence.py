import torch
import matplotlib.pyplot as plt
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.utils.torch_utils import max0, N_cdf
from torch.autograd.functional import jvp
from tqdm import tqdm


if __name__ == '__main__':

    N_train = 1024 * 8  # Number of training samples
    N_test = 512        # Number of test samples
    hedge_steps = torch.arange(start=1, end=251, step=1)

    use_analyic_delta = False
    restrict_delta = False      # Force delta to be between 0 and 1

    use_av = True
    scale_training_range_each_step = True

    r = torch.tensor(0.03)
    sigma = torch.tensor(0.2)
    T = torch.tensor(1.0)
    S0 = torch.tensor(100.0)
    K = torch.tensor(100.0)

    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=True)
    scalar = DifferentialStandardScaler()

    s_train_vec = torch.linspace(20.0, 180, N_train)

    def simulate(spot, vol, rate, expiry, Z):
        return spot * torch.exp((rate - 0.5 * vol ** 2) * expiry - torch.sqrt(expiry) * vol * Z)

    def bs_call(spot, vol, rate, expiry, strike):
        d1 = (torch.log(spot / strike) + (rate + 0.5 * vol ** 2) * expiry) / (vol * torch.sqrt(expiry))
        d2 = d1 - vol * torch.sqrt(expiry)
        return N_cdf(d1) * spot - N_cdf(d2) * strike * torch.exp(-rate * expiry)

    def bs_delta(spot, vol, rate, expiry, strike):
        d1 = (torch.log(spot / strike) + (rate + 0.5 * vol ** 2) * expiry) / (vol * torch.sqrt(expiry))
        return N_cdf(d1)

    def calc_dCds(spot_vec, tau):
        def payoff(s0):
            rv = torch.randn((len(spot_vec),))
            ST = simulate(s0, sigma, r, tau, rv)
            df = torch.exp(-r * tau)
            return df * max0(ST - K)
        ones = torch.ones_like(spot_vec)
        return jvp(payoff, spot_vec, ones, create_graph=False)

    def training_data(spot_vec: torch.Tensor, tau: float = 0.0, use_av: bool = True):
        if use_av:
            # X_train[i] = X_train[i + N_train],  for all i, when using AV
            spot_vec = torch.concat([spot_vec, spot_vec])

        y, dydr = calc_dCds(spot_vec, tau)

        X_train = spot_vec.reshape(-1, 1)
        y_train = y.reshape(-1, 1)
        z_train = dydr.reshape(-1, 1)

        if use_av:
            idx_half = N_train
            X_train = X_train[:idx_half]
            y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
            z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])

        return X_train, y_train, z_train

    def calc_delta(spot_vec: torch.Tensor, s_train_vec: torch.Tensor, tau, use_av: bool = True) -> torch.Tensor:
        X_test = spot_vec.reshape(-1, 1)

        X_train, y_train, z_train = training_data(s_train_vec, tau, use_av)

        X_train_scaled, y_train_scaled, z_train_scaled = scalar.fit_transform(X_train, y_train, z_train)

        diff_reg.fit(X_train_scaled, y_train_scaled, z_train_scaled)

        X_test_scaled, _, _ = scalar.transform(X_test, None, None)
        y_pred_scaled, z_pred_scaled = diff_reg.predict(X_test_scaled, predict_derivs=True)

        _, y_pred, z_pred = scalar.predict(None, y_pred_scaled, z_pred_scaled)

        y_scaled, z_scaled = diff_reg.predict(X_train_scaled, predict_derivs=True)
        _, y, z = scalar.predict(None, y_scaled, z_scaled)

        return z_pred.flatten()

    # Price of call option
    eu_call = bs_call(S0, sigma, r, T, K)

    pnl_list = []

    for M in tqdm(hedge_steps, desc='Performing hedging experiments'):

        TL = torch.linspace(0.0, float(T), M + 1)
        dt = T / M

        G = torch.randn((M, N_test))

        # Initialize
        S = torch.full((M + 1, N_test), torch.nan)
        S[0] = S0

        V = eu_call * torch.ones((N_test, ))
        if use_analyic_delta:
            # Note, using the average af that is the estimator in a single point
            h_a = bs_delta(S[0, :], vol=sigma, rate=r, expiry=T, strike=K)
        else:
            h_a = torch.mean(calc_dCds(S[0, :], T)[1])

        h_b = V - h_a * S[0, :]

        for j, t in enumerate(TL[1:], start=1):
            S[j] = simulate(S[j-1], sigma, r, dt, G[j-1])
            V = h_a * S[j, :] + h_b * torch.exp(r * dt)
            if j < M:
                if scale_training_range_each_step:
                    sd = torch.std(S[j, :]) * torch.sqrt(T-t)
                    s_train_vec = torch.linspace(S[j, :].min() - sd, S[j, :].max() + sd, N_train)
                if use_analyic_delta:
                    h_a = bs_delta(S[j, :], vol=sigma, rate=r, expiry=T-t, strike=K)
                else:
                    h_a = calc_delta(spot_vec=S[j, :], s_train_vec=s_train_vec, tau=T-t)
                    if restrict_delta:
                        torch.minimum(torch.maximum(h_a, torch.ones_like(S[j, :])), torch.zeros_like(S[j, :]))
                h_b = V - h_a * S[j, :]

        ST = S[-1, :]
        payoff = max0(ST - K)
        pnl = payoff - V

        pnl_list.append(pnl)

    pnl_mean = torch.concat([torch.mean(pnl).view(1) for pnl in pnl_list])
    pnl_std = torch.concat([torch.std(pnl).view(1) for pnl in pnl_list])

    fig, ax = plt.subplots(2, sharex='all')
    ax[0].plot(hedge_steps, pnl_mean, color='black')
    ax[1].plot(hedge_steps, pnl_std, color='black')

    ax[0].set_ylabel('mean(PnL)')
    ax[1].set_ylabel('std(PnL)')
    ax[1].set_xlabel('Number of hedge points')

    ax[1].set_ylim(-0.1, torch.max(pnl_std) * 1.05)
    plt.show()



