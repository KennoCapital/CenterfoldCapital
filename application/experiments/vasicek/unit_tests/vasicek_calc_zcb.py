from application.engine.vasicek import Vasicek
import matplotlib.pyplot as plt
import torch

"""checking term structure of vasicek model"""
if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.02)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor([0.02, 0.03, 0.04])

    t = torch.tensor([0.25, 0.50, 0.75])

    model = Vasicek(a, b, sigma, r0)

    zcb = model.calc_zcb(r0, t)
    print(zcb)

    T = torch.linspace(0.25, 20, 50)
    zcbs = model.calc_zcb(torch.tensor(0.08), T)
    y = - torch.log(zcbs).reshape(-1) * (1/T)

    plt.figure()
    plt.plot(T, zcbs, 'r-')
    plt.title('term structure T --> P(0,T)')
    plt.show()

