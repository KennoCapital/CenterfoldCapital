import numpy as np


def sim_euler(t, W, x, a, b):
    """
    Function for simulating Euler discretization of an Ito-process of the form
            dX(t) = a(X(t), t) * dt + b(X(t), t) * dW(t)
    :param t:   Time steps (including start and end points)
    :param W:   Wiener process
    :param x:   Underlying process (must contain initial values in the first row)
    :param a:   Drift function
    :param b:   Diffusion function

    :return:    Simulated values of the underlying process
    """
    dt = np.diff(t)
    dW = np.diff(W, axis=0)
    for j, s in enumerate(t[1:]):
        x[j + 1, :] = x[j, :] + a(x[j, :], s) * dt[j] + b(x[j, :], s) * dW[j, :]
    return x


def sim_milstein(t, W, x, a, b):
    """
    Function for performing Milstein simulation of SDE using Runge-Kutta.

    :param t:   Time steps (including start and end points)
    :param W:   Wiener process
    :param x:   Underlying process (must contain initial values in the first row)
    :param a:   Drift function
    :param b:   Diffusion function

    :return:    Simulated values of the underlying process
    """
    dt = np.diff(t)
    dW = np.diff(W, axis=0)
    for j, s in enumerate(t[1:]):
        x_hat = x[j, :] + a(x[j, :], s) * dt[j] + b(x[j, :], s) * np.sqrt(dt[j])

        x[j + 1] = x[j, :] + a(x[j, :], s) * dt[j] + b(x[j, :], s) * dW[j, :] + \
                   1 / (2 * np.sqrt(dt[j])) * (dW[j, :] ** 2 - dt[j]) * (b(x_hat, s) - b(x[j, :], s))
    return x


class WienerProcess:
    def __init__(self, t, N, use_av=True, seed=None):
        """
        :param t:       Time steps (including start and end points)
        :param N:       Number of simulations
        :param use_av:  Use antithetic variates to reduce variance
        :param seed:    Seed for replicating results
        """
        self.t = t
        self.dt = np.diff(t)
        self.N = N
        self.M = len(t) - 1
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.use_av = use_av
        self.Z = None
        self.W = None
        self.dW = None
        self.t_B = None
        self.B = None

    def sim_wienerprocess(self):
        if self.use_av:
            if self.N % 2 != 0:
                raise ValueError('N (={}) must be even when using antithetic variates'.format(self.N))
            self.Z = self.rng.standard_normal(size=(self.M, self.N // 2))
            self.Z = np.hstack([self.Z, -self.Z])
        else:
            self.Z = self.rng.standard_normal(size=(self.M, self.N))

        self.dW = self.Z * np.sqrt(self.dt.reshape(-1, 1))
        self.W = np.cumsum(np.vstack([np.zeros(shape=(1, self.N)), self.dW]), axis=0)
        return self.W

    def sim_brownian_bridge(self, t, W_t, W_T, rng=None):
        """
        Simulate brownian bridge.
        :param t:           Time points
        :param W_t_         Initial value of brownian motion
        :param W_T          Terminal value of brownian motion
        :param rng:         numpy.random.Generator
        :return:            Simulated brownian bridge

        """
        self.t_B = t
        dt = np.diff(t)
        M = len(t) - 1

        if rng is None:
            rng = self.rng

        if self.use_av:
            Z = rng.standard_normal(size=(M, N // 2))
            Z = np.hstack([Z, Z])
        else:
            Z = rng.standard_normal(size=(M, N))

        self.B = np.zeros(shape=(M + 1, N))
        self.B[0] = W_t

        for j in range(M):
            mu = W_T * (M - 1) / (M - 1 + 1)
            sigma = np.sqrt(dt[j] * (M - j) / (M - j + 1))
            self.B[j + 1] = self.B[j] + mu * np.sqrt(sigma) * Z[j]
        return self.B

    def bridge_brownian_motion(self):
        """
        Combine a brownian motion and its bridge to get a more granular simulation
        :param t_W: Time points of the brownian motion
        :param t_B: Time points of the brownian bridge
        :param W:   Values of the brownian motion
        :param B:   Values of the brownian bridge
        :return:    The filled timeline and filled brownian motion (both sorted by time)
        """
        self.t = np.concatenate([self.t, self.t_B[1:-1]])
        order = np.argsort(self.t)
        self.W = np.vstack([self.W, self.B])[order, :]
        self.t = self.t[order]
        return self.t, self.W


class GBM(WienerProcess):
    def __init__(self, t, x0, N, mu, sigma, use_av=True, seed=None):
        """
        :param t:       Time steps (including start and end points)
        :param x0:      Inital value(s)
        :param N:       Number of simulationg
        :param mu:      Drift coefficient(s)
        :param sigma:   Volatility coefficient(s)
        :param use_av:  Use antithetic variates to reduce variance
        :param seed:    Seed for replication
        """
        super().__init__(t=t, N=N, use_av=use_av, seed=seed)
        self.x0 = x0
        self.mu = mu
        self.sigma = sigma

        self.X = np.full(shape=(len(t), N), fill_value=np.nan)
        self.X[0, :] = x0

    def sim_exact(self):
        """
        Method for simulating Geometric Brownian Motion over specified time steps and initial values of the processes.
        The simulation uses the discretized analytical solution,
            X(t+1) = X(t) * exp{(mu-0.5*sigma^2)*dt + sigma * W(t)}
        :return:
        """
        if self.W is None:
            super().sim_wienerprocess()
        self.X = self.x0 * np.exp((self.mu - 0.5 * self.sigma ** 2) * self.t.reshape(-1, 1) + self.sigma * self.W)
        return self.X

    def sim_euler(self):
        """
        Method for simulating Euler discretization of an Ito-process of the form
            dX(t) = a(X(t), t) * dt + b(X(t), t) * dW(t)
        """
        if self.W is None:
            super().sim_wienerprocess()
        self.X = sim_euler(t=self.t, W=self.W, x=self.X, a=self._a, b=self._b)
        return self.X

    def sim_milstein(self):
        """
        Method for performing Milstein simulation of SDE using Runge-Kutta
        :return:
        """
        if self.W is None:
            super().sim_wienerprocess()
        self.X = sim_milstein(t=self.t, W=self.W, x=self.X, a=self._a, b=self._b)
        return self.X

    def _a(self, x, t):
        return self.mu * x

    def _b(self, x, t):
        return self.sigma * x


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Parameters
    t0 = 0.0
    T = 1.0
    x0 = 50
    M = 50
    N = 100
    mu = 0.07
    sigma = 0.2
    seed = 1
    use_av = True

    # Equidistant time steps
    t = np.linspace(start=t0, stop=T, num=M + 1, endpoint=True)

    # Simulation
    S = GBM(t=t, x0=x0, N=N, mu=mu, sigma=sigma, use_av=use_av, seed=seed).sim_exact()

    plt.plot(t, S)
    plt.show()
