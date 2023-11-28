from application.engine.vasicek import Vasicek
import torch

torch.set_default_dtype(torch.float64)


if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)

    measure = 'risk_neutral'

    mdl = Vasicek(a=a, b=b, sigma=sigma, r0=r0, use_ATS=False, use_euler=False, measure=measure)

    delta = torch.tensor(0.25)
    swapFirstFixingDate = torch.tensor(1.0)
    swapLastFixingDate = torch.tensor(6.0)
    notional = torch.tensor(1E6)
    t = torch.linspace(float(swapFirstFixingDate),
                       float(swapLastFixingDate),
                       int((swapLastFixingDate - swapFirstFixingDate) / delta + 1))

    swap_rate = mdl.calc_swap_rate(r0, t, delta)
    swap = mdl.calc_swap(r0, t, delta, swap_rate, notional)
