import torch
from application.engine.vasicek import Vasicek
from application.utils.torch_utils import N_cdf
from matplotlib import pyplot as plt

'''
    Source: Björk prop. 21.10 (page 293)
    Value and delta of an European call (expiry T) on a Zero Coupon Bond with maturity T + delta 
'''

if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    mdl = Vasicek(a, b, sigma, r0)

    def calc_call(zcb1, zcb2, t, delta, K):
        vol = sigma ** 2 / (2 * a ** 3) * (1 - torch.exp(-2 * a * t)) * (1 - torch.exp(-a * delta)) ** 2
        d2 = (torch.log(zcb2 / (K * zcb1)) - 0.5 * vol) / torch.sqrt(vol)
        d1 = d2 + torch.sqrt(vol)
        return zcb2 * N_cdf(d1) - K * zcb1 * N_cdf(d2)

    def calc_delta(zcb1, zcb2, t, delta, K):
        vol = sigma ** 2 / (2 * a ** 3) * (1 - torch.exp(-2 * a * t)) * (1 - torch.exp(-a * delta)) ** 2
        d2 = (torch.log(zcb2 / (K * zcb1)) - 0.5 * vol) / torch.sqrt(vol)
        d1 = d2 + torch.sqrt(vol)
        return N_cdf(d1)

    T = torch.tensor(5.0)
    K = torch.tensor(0.70)

    delta = torch.linspace(0.0, 30.0, 1001)
    delta = 1.0

    zcb1 = mdl.calc_zcb(r0, T).flatten()
    zcb2 = torch.linspace(0.0, 1.0, 1001)  # mdl.calc_zcb(r0, T+t).flatten()
    call_value = calc_call(zcb1, zcb2, T, T + delta, K)
    call_delta = calc_delta(zcb1, zcb2, T, T + delta, K)

    fig, ax = plt.subplots(2, sharex='all')
    ax[0].plot(zcb2, call_value)
    ax[0].set_ylabel('Value of Call')

    ax[1].plot(zcb2, call_delta)
    ax[1].set_xlim(K * zcb1 - 0.05, K * zcb1 + 0.05)
    ax[1].set_ylabel('Delta of Call')
    ax[1].set_xlabel('P(0,T+delta)')
    fig.suptitle(f'Call option of ZCB(0, T+delta) with strike {K:.2f}')
    plt.show()
