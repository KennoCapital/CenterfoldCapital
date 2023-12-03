import torch
import matplotlib.pyplot as plt
from application.engine.vasicek import Vasicek


if __name__ == '__main__':
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    measure = 'risk_neutral'
    mdl = Vasicek(a, b, sigma, r0, use_ATS=True, use_euler=False, measure=measure)

    delta = torch.tensor(0.25)

    swap_curve = []
    maturity = []
    for i in range(1, 121):
        T = i * delta
        fixings = torch.linspace(float(delta), T, i)
        swap_rate = mdl.calc_swap_rate(r0, fixings, delta)

        maturity.append(T.view(1))
        swap_curve.append(swap_rate)

    maturity = torch.concat(maturity)
    swap_curve = torch.concat(swap_curve)


    plt.plot(maturity, swap_curve, color='blue')
    plt.title('Swap Rate Curve')
    plt.xlabel('Maturity ($T_n$)')
    plt.ylabel('Swap Rate')
    plt.show()
