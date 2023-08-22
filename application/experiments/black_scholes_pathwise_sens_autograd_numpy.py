import autograd.numpy as np
from autograd import jacobian
import matplotlib.pyplot as plt


def payoff_call(x, strike):
    return np.maximum(x - strike, 0.0)


def payoff_id(x, strike):
    return x


def sim_gbm(N, t, spot, strike, drift, vol, func, seed=None):
    M = len(t)
    dt = np.diff(t, axis=0)
    rng = np.random.default_rng(seed=seed)
    Z = rng.normal(loc=0.0, scale=1.0, size=(M-1, N))

    W = np.cumsum(np.concatenate([
        np.zeros(shape=(1, N)), np.sqrt(dt) * Z
    ]), axis=0)

    S = spot * np.exp((drift - 0.5 * vol ** 2) * t + vol * W)
    return func(S, strike)


if __name__ == '__main__':
    seed = 1234
    N = 1000
    M = 52
    t0 = 0.0
    T = 1.0
    spot = 100.0
    strike = 100.0
    drift = 0.03
    vol = 0.2

    t = np.linspace(t0, T, M+1, True).reshape(-1, 1)

    from datetime import datetime

    start = datetime.now()
    S = sim_gbm(N, t, spot, strike, drift, vol, payoff_id, seed)
    stop = datetime.now()
    print(stop - start)

    start = datetime.now()
    dCdS = jacobian(sim_gbm, argnum=2)(N, t, spot, strike, drift, vol, payoff_call, seed)
    stop = datetime.now()
    print(stop - start)

    plt.scatter(S, dCdS, color='gray', alpha=.1)
    plt.show()


