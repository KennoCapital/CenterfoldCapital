from abc import ABC, abstractmethod
from products import Portfolio
import torch



N = lambda x: torch.distributions.Normal(loc=0.0, scale=1.0).cdf(x)


class MCModel(ABC):
    @abstractmethod
    def __init__(self, portfolio: Portfolio):
        self.timeline = portfolio.timeline

    @abstractmethod
    def make_timeline(self):
        pass

    @abstractmethod
    def generatePaths(self):
        pass


class VasicekMCModel(MCModel):
    def __init__(self,
                 timeline:  torch.Tensor,
                 portfolio: Portfolio,
                 a:         float,
                 b:         float,
                 sigma:     float):

        self.timeline = timeline
        self.portfolio = portfolio
        self.a = a
        self.b = b
        self.sigma = sigma

    def generatePaths(self, gaussVec):
        pass


class Vasicek:
    def __init__(self, a, b, sigma):
        self.a = a
        self.b = b
        self.sigma = sigma

    def calc_inst_fwd_rate(self, t, r0):
        """f(0,t)"""
        return r0 * torch.exp(-self.a * t) + \
               self.b * (1-torch.exp(-self.a * t)) - \
               self.sigma**2 * (1 - torch.exp(-self.a * t))**2 / (2*self.a**2)

    def _calc_A(self, t):
        B = self._calc_B(t)
        return (self.b - self.sigma**2 / (2*self.a**2)) * (t - B) - \
               self.sigma ** 2 * B ** 2 / (4 * self.a)

    def _calc_B(self, t):
        return  (1 - torch.exp(-self.a * t)) / self.a

    def calc_zcb_AB(self, t, r0):
        """P(0,t)=exp{ -A(0,t) - B(0,t) * r(t)}"""
        return torch.exp(-self._calc_A(t) - self._calc_B(t) * r0)

    def calc_zcb(self, t, r0):
        """P(0,t)"""
        return torch.exp(
            (r0 - self.b) * (torch.exp(-self.a * t) - 1) / self.a - \
            self.b * t + self.sigma**2 * t / (2 * self.a ** 2) + \
            self.sigma ** 2 * (4 * torch.exp(-self.a * t) - torch.exp(-2 * self.a * t) - 3) / (4 * self.a**3)
        )

    def calc_fwd_rate(self, t, dt, r0):
        """F(0,t,t+dt)"""
        zcb_t = self.calc_zcb(t, r0)
        zcb_tdt = self.calc_zcb(t+dt, r0)
        return (zcb_t / zcb_tdt - 1) / dt

    def calc_swap_rate(self, t, dt, r0):
        zcb_0 = self.calc_zcb(t[0], r0)
        zcb_T = self.calc_zcb(t[-1], r0)
        return (zcb_0 - zcb_T) / (dt * torch.sum(self.calc_zcb(t[1:], r0)))

    def calc_caplet_price_bond_options(self, t, dt, r0, K):
        zcb_t = self.calc_zcb(t, r0)
        zcb_tdt = self.calc_zcb(t+dt, r0)
        K_bar = 1 / (1 + dt * K)

        integral = self.sigma**2 / self.a**3 * \
                   (0.5 * (torch.exp(2*t) + torch.exp(2*(t+dt)) - torch.exp(2*dt)) + torch.exp(dt) - torch.exp(2*t+dt))
        d1 = (torch.log(zcb_t * K_bar / zcb_tdt)) + 0.5 * integral / torch.sqrt(integral)
        d2 = d1 - torch.sqrt(integral)
        return zcb_t * N(d1) - zcb_tdt / K_bar * N(d2)

    def calc_caplet_price_black(self, t, dt, r0, K):
        """Eq. 2.6, Filipovic"""
        fwd = self.calc_fwd_rate(t, dt, r0)
        zcb = self.calc_zcb(t, r0)
        d1 = (torch.log(fwd / K) + 0.5 * self.sigma ** 2 * t) / (self.sigma * torch.sqrt(t))
        d2 = (torch.log(fwd / K) - 0.5 * self.sigma ** 2 * t) / (self.sigma * torch.sqrt(t))
        return dt * zcb * ( fwd * N(d1) - K * N(d2) )

    def calc_cap_price_black(self, t, dt, r0, K):
        # cpl = self.calc_caplet_price_bond_options(t, dt, r0, K)
        cpl = self.calc_caplet_price_black(t, dt, r0, K)
        return torch.sum(cpl)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    dt = torch.tensor(0.25)
    K = torch.tensor(0.08)

    t = torch.linspace(0.25, 2.0, 8)

    mld = Vasicek(a=a, b=b, sigma=sigma)

    inst_fwd = mld.calc_inst_fwd_rate(t=t, r0=r0)

    zcb = mld.calc_zcb(t=t, r0=r0)

    fwd = mld.calc_fwd_rate(t=t, dt=dt, r0=r0)

    cpl = mld.calc_caplet_price_black(t=t, dt=dt, r0=r0, K=K)

    swap_rate = mld.calc_swap_rate(t=t, dt=dt, r0=r0)

    cap = mld.calc_cap_price_black(t=t, dt=dt, r0=r0, K=swap_rate)
    print(cap*100)

    print(
        mld.calc_caplet_price_black(t=torch.tensor(0.25), dt=torch.tensor(0.25), r0=r0, K=swap_rate),
        mld.calc_caplet_price_bond_options(t=torch.tensor(0.25), dt=torch.tensor(0.25), r0=r0, K=swap_rate)
    )

    # Manual
    zcb_1 = mld.calc_zcb(t=torch.tensor(0.25), r0=r0)
    zcb_2 = mld.calc_zcb(t=torch.tensor(0.50), r0=r0)
    zcb_3 = mld.calc_zcb(t=torch.tensor(0.75), r0=r0)
    zcb_4 = mld.calc_zcb(t=torch.tensor(1.00), r0=r0)
    swap_rate = (zcb_1 - zcb_4) / (0.25 * (zcb_2 + zcb_3 + zcb_4))
    fwd_1_2 = (zcb_1 / zcb_2 - 1) / 0.25
    fwd_2_3 = (zcb_2 / zcb_3 - 1) / 0.25
    fwd_3_4 = (zcb_3 / zcb_4 - 1) / 0.25

    d1_1_2 = (torch.log(fwd_1_2 / swap_rate) + 0.5 * sigma * 0.25) / (sigma * 0.25)
    d2_1_2 = (torch.log(fwd_1_2 / swap_rate) - 0.5 * sigma * 0.25) / (sigma * 0.25)
    d1_2_3 = (torch.log(fwd_2_3 / swap_rate) + 0.5 * sigma * 0.50) / (sigma * 0.50)
    d2_2_3 = (torch.log(fwd_2_3 / swap_rate) - 0.5 * sigma * 0.50) / (sigma * 0.50)
    d1_3_4 = (torch.log(fwd_3_4 / swap_rate) + 0.5 * sigma * 0.75) / (sigma * 0.75)
    d2_3_4 = (torch.log(fwd_3_4 / swap_rate) - 0.5 * sigma * 0.75) / (sigma * 0.75)
    cpl_1_2 = 0.25 * zcb_2 * (fwd_1_2 * N(d1_1_2) - swap_rate * N(d2_1_2))
    cpl_2_3 = 0.25 * zcb_3 * (fwd_2_3 * N(d1_2_3) - swap_rate * N(d2_2_3))
    cpl_3_4 = 0.25 * zcb_4 * (fwd_3_4 * N(d1_3_4) - swap_rate * N(d2_3_4))
    cap = cpl_1_2 + cpl_2_3 + cpl_3_4
    print(cap)

    # TODO this is not working: Filipovic table 7.1
    """
    mat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0])

    for m in mat:
        timeline = torch.linspace(0.25, m, steps=int(4 * m))
        swap_rate = mld.calc_swap_rate(t=timeline, dt=dt, r0=r0)
        cap = mld.calc_cap_price_black(t=timeline, dt=dt, r0=r0, K=swap_rate)
        print(f'Maturity = {m},\t swap_rate = {swap_rate},\t ATM-cap price = {cap}')
        plt.scatter(m, cap)
    plt.show()
    """