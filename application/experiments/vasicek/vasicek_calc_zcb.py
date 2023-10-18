from application.engine.vasicek import Vasicek
import torch


if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.02)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor([0.02, 0.03, 0.04])

    t = torch.tensor([0.25, 0.50, 0.75])

    model = Vasicek(a, b, sigma, r0)

    zcb = model.calc_zcb(r0, t)
    print(zcb)
