import torch
from torch.autograd.functional import jacobian


def sim_gbm(N, t, spot, drift, vol, seed=None):
    M = int(t.shape[0])
    dt = torch.diff(t)

    Z = torch.normal(mean=0.0, std=1.0, size=(M - 1, N), generator=torch.manual_seed(seed))

    W = torch.concatenate([
        torch.zeros(size=(1, N)), torch.sqrt(dt)[:, None] * Z
    ]).cumsum(axis=0)

    S = spot * torch.exp(((drift - 0.5 * vol ** 2) * t)[:, None] + vol * W)
    return S[-1,]


def gbm_wrapper(spot, vol):
    return sim_gbm(N, t, spot, drift, vol, seed)


if __name__ == '__main__':
    from datetime import datetime
    seed = 1234
    N = 1000
    M = 52
    t0 = 0.0
    T = 1.0
    spot = torch.tensor(100.0, requires_grad=True)
    drift = torch.tensor(0.03, requires_grad=True)
    vol = torch.tensor(0.2, requires_grad=True)
    t = torch.linspace(t0, T, M + 1)

    start = datetime.now()

    J = jacobian(func=gbm_wrapper, inputs=(spot, vol))
    stop = datetime.now()
    print(stop - start)
