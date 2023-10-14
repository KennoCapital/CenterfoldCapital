from application.engine.vasicek import Vasicek
import torch


if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.02)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.02)

    start = torch.tensor(0.25)
    delta = torch.tensor(0.25)

    model = Vasicek(a, b, sigma, r0)

    strike = model.calc_swap_rate(r0, start.view(1), delta)

    cpl = model.calc_cpl(r0, start, delta, strike)

    print(cpl * 1e6)

