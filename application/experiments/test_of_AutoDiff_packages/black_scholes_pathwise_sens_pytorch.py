import torch
from torch.autograd.functional import jacobian

class simulator:
    def __init__(self, N, M, t0, T, spot, r, strike, drift, vol, seed = None):
        self.N = N
        self.M = M
        self.t0 = t0
        self.T = T
        self.r = r
        self.spot = spot
        self.strike = strike
        self.drift = drift
        self.vol = vol
        self.seed = seed
        self.t = torch.linspace(self.t0, self.T, self.M + 1, dtype=torch.float64)

    def sim_gbm(self):
        N, t, spot, drift, vol = self.N, self.t, self.spot, self.drift, self.vol
        M = int(t.shape[0])
        dt = torch.diff(t)

        Z = torch.normal(mean=0.0, std=1.0, size=(M - 1, N), generator=torch.manual_seed(self.seed))

        W = torch.cat([torch.zeros(size=(1, self.N)), torch.sqrt(dt)[:, None] * Z]).cumsum(axis=0)
        # torch.concatenate([torch.zeros(size=(1, N)), torch.sqrt(dt)[:, None] * Z]).cumsum(axis=0)

        S = self.spot * torch.exp(((self.drift - 0.5 * self.vol ** 2) * self.t)[:, None] + self.vol * W)
        return S[-1,]

    def payoff(self, S_T):
        C = torch.clamp(S_T - self.strike, min=0)
        return C


def compute_jacobian(obj, spot, vol):
    def wrapper(spot, vol):
        obj.spot = spot
        obj.vol = vol
        S_T = obj.sim_gbm()
        return obj.payoff(S_T)

    return jacobian(wrapper, (spot, vol), create_graph=False, strategy="reverse-mode")

if __name__ == '__main__':
    seed = 1234
    N = 4
    M = 52
    t0 = 0.0
    T = 1.0
    r = torch.tensor(0.05)
    spot = torch.tensor(100.0, requires_grad=True)
    strike = torch.tensor(100.0)
    drift = torch.tensor(0.03)
    vol = torch.tensor(0.2, requires_grad=True)
    t = torch.linspace(t0, T, M + 1)

    obj = simulator(N=N, M=M, t0=t0, T=T, spot=spot, r=r, strike=strike, drift=drift, vol=vol, seed=seed)
    #J_class = jacobian(obj.payoff, (spot, vol), create_graph=True, strategy="reverse-mode")
    J_class = compute_jacobian(obj, spot, vol)

    print("class delta: ", J_class[0].detach().numpy())
    print("class delta: ", J_class[1].detach().numpy())