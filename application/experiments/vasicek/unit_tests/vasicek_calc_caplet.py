from application.engine.vasicek import Vasicek
import torch

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=8)


if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor([0.08])

    start = torch.tensor([0.25, 0.50, 0.75])
    delta = torch.tensor([0.25])

    model = Vasicek(a, b, sigma, r0)

    strike = model.calc_swap_rate(r0, start, delta)

    cpl = model.calc_cpl(r0, start, delta, strike)

    print(torch.sum(cpl))  # 0.00215686

