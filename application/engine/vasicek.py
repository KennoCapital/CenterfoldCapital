import torch


def torch_rng(seed=None):
    if seed is None:
        return torch.Generator().manual_seed(torch.Generator().initial_seed())
    else:
        return torch.Generator().manual_seed(seed)


def sim_short_rate_vasicek_euler(t, N, r0, a, b, sigma, seed=None):
    """
    p. 110 - Glasserman (2003)
    :param t:
    :param N:
    :param r0:
    :param a:
    :param b:
    :param sigma:
    :param seed:
    :return:
    """
    M = len(t) - 1
    r = torch.full(size=(M + 1, N), fill_value=torch.nan)
    r[0, :] = r0

    rng = torch_rng(seed=seed)

    for j in range(M):
        dt = t[j + 1] - t[j]
        Z = torch.randn(size=(1, N), generator=rng)
        r[j + 1, :] = a * (b - r[j, :]) * dt + sigma * torch.sqrt(dt) * Z

    return r


def sim_short_rate_vasicek_exact(t, N, r0, a, b, sigma, seed=None):
    """
    Eq.(3.46), p. 110 - Glasserman (2003)
    :param t:
    :param N:
    :param r0:
    :param a:
    :param b:
    :param sigma:
    :param seed:
    :return:
    """
    M = len(t) - 1
    r = torch.full(size=(M + 1, N), fill_value=torch.nan)
    r[0, :] = r0

    rng = torch_rng(seed=seed)

    for j in range(M):
        dt = t[j + 1] - t[j]
        Z = torch.randn(size=(1, N), generator=rng)
        r[j + 1, :] = torch.exp(-a * dt) * r[j] + \
                      b * (1 - torch.exp(-a * dt)) + \
                      sigma * Z * torch.sqrt(1 / (2 * a) * (1 - torch.exp(-2 * a * dt)))

    return r


def zcb_vasicek()




if __name__ == '__main__':
    seed = 1
    t0 = 0.0
    T = 2.0
    a = 1.0
    b = 0.03
    r0 = 0.03
    sigma = 0.2
    N = 100000
    M = int(50 * T)
    K = 0.03

    t = torch.linspace(start=t0, end=T, steps=M + 1)

    r_euler = sim_short_rate_vasicek_euler(t, N, r0, a, b, sigma, seed)
    r_exact = sim_short_rate_vasicek_exact(t, N, r0, a, b, sigma, seed)

    # ------------------------------------------------- #
    # 1) r(t)   // Sim this for t=T
    # 2) P(t,T) = e^{-A(t,T) - B(t,T)*r(t)}
    # 3) F(t, T, T+dt)
    # 4) cpl(t) = E^Q_t [ max{ F(T, T, T+dt) - K ,0.0 } ]
    # ------------------------------------------------- #

