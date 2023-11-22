import torch
import matplotlib.pyplot as plt
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.utils.torch_utils import max0, N_cdf
from torch.autograd.functional import jvp


if __name__ == '__main__':

    N_train = 1024 * 8 # Number of training samples
    N_test = 256    # Number of test samples
    M = 10          # Number of hedge points

    use_av = True
    scale_training_range_each_step = True

    r = torch.tensor(0.03)
    sigma = torch.tensor(0.2)
    T = torch.tensor(0.25)
    S0 = torch.tensor(100.0)
    K = torch.tensor(100.0)

    deg = 5
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=True)
    scalar = DifferentialStandardScaler()

    TL = torch.linspace(0.0, float(T), M + 1)
    dt = T / M

    G = torch.randn((M, N_test))

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

    def calc_dCds(spot_vec, tau, use_av: bool=True):
        def payoff(s0):
            if use_av:
                rv = torch.randn((len(spot_vec)//2,))
                rv = torch.concat([rv, -rv])
            else:
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

        y, dydr = calc_dCds(spot_vec, tau, use_av)

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

        fig, ax = plt.subplots(2, sharex='all')
        ax[0].plot(X_train, y_train, 'o', color='gray', alpha=0.50)
        ax[0].plot(X_train, y, color='blue', alpha=0.25)
        ax[0].plot(X_test, y_pred, 'o', color='orange', alpha=0.25)
        ax[0].plot(X_train, bs_call(X_train, sigma, r, tau, K), color='black')
        ax[0].set_ylabel('Value')
        ax[1].plot(X_train, z_train, 'o', color='gray', label='Pathwise Samples', alpha=0.25)
        ax[1].plot(X_train, z, color='blue', label='In-sample prediction', alpha=0.50)
        ax[1].plot(X_test, z_pred, 'o', color='orange', label='Out-of-sample prediction', alpha=0.25)
        ax[1].plot(X_train, bs_delta(X_train, sigma, r, tau, K), label='Black-Scholes', color='black')
        ax[1].set_ylabel('Delta')
        ax[1].set_xlabel('S(t)')

        # Adjust size of plot
        box = ax[0].get_position()
        ax[0].set_position([box.x0, box.y0, box.width, box.height * 0.8])
        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width, box.height * 0.8])

        # Legend
        handles, labels = fig.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, fancybox=True, shadow=True,
                   bbox_to_anchor=(0.5, 0.90))

        plt.show()

        return z_pred.flatten()


    # Price of call option
    eu_call = bs_call(S0, sigma, r, T, K)

    # Initialize
    S = torch.full((M + 1, N_test), torch.nan)
    S[0] = S0

    V = eu_call * torch.ones((N_test, ))
    h_a = torch.mean(calc_dCds(S[0, :], T)[1])  # Note, using the average af that is the estimator in a single point
    h_b = V - h_a * S[0, :]

    for j, t in enumerate(TL[1:], start=1):
        S[j] = simulate(S[j-1], sigma, r, dt, G[j-1])
        V = h_a * S[j, :] + h_b * torch.exp(r * dt)
        if j < M:
            if scale_training_range_each_step:
                sd = torch.std(S[j, :]) * torch.sqrt(T-t)
                s_train_vec = torch.linspace(S[j, :].min() - sd, S[j, :].max() + sd, N_train)
            h_a = calc_delta(spot_vec=S[j, :], s_train_vec=s_train_vec, tau=T-t)
            h_b = V - h_a * S[j, :]

    ST = S[-1, :]
    payoff = max0(ST - K)

    x = torch.linspace(ST.min(), ST.max(), N_test)
    y = max0(x - K)

    plt.figure()
    plt.plot(ST, V, 'o', color='orange', alpha=0.25)
    plt.plot(x, y, color='black')
    plt.title('Black-Scholes: Replication of European Call Payoff Function')
    plt.xlabel('S(T)')
    plt.ylabel('Payoff')
    plt.show()
