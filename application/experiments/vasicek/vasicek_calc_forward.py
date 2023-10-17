from application.engine.vasicek import Vasicek
from application.engine.linearProducts import forward
import torch


if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.02)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor([0.02, 0.03, 0.04])

    t = torch.tensor([0.25, 0.50, 0.75])
    delta = torch.tensor([0.25, 0.50, 0.75])  #

    model = Vasicek(a, b, sigma, r0)

    fwd = model.calc_fwd(r0, t, delta)

    print(fwd)

    # Test implementation manually
    zcb_t = model.calc_zcb(r0, t)
    zcb_tdt = model.calc_zcb(r0, t + delta)

    fwd2 = torch.full_like(zcb_t, torch.nan)
    for i in range(len(zcb_t)):
        for j in range(len(zcb_t[i])):
            fwd2[i, j] = forward(zcb_t[i, j], zcb_tdt[i, j], delta[i])

    print(fwd2)
