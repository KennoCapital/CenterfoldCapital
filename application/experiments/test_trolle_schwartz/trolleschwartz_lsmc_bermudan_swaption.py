from application.engine.mcBase import lsmcDefaultSim, LSMC, RNG, mcSim
from application.engine.products import BermudanPayerSwaption, EuropeanPayerSwaption
from application.engine.trolleSchwartz import trolleSchwartz
from application.engine.regressor import PolynomialRegressor
import torch
import matplotlib.pyplot as plt

torch.set_printoptions(2)
torch.set_default_dtype(torch.float64)

"""
Testing implementation of Bermudan swaption and computes naive upper and lower bound.
"""

if __name__ == '__main__':
    seed = 1234
    plot_lower_bound = True

    deg = 5
    n = 500
    N = 5000

    measure = 'risk_neutral'

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.5509)  # 0553
    sigma = torch.tensor(1.0497)  # 3325
    alpha0 = torch.tensor(0.000)  # 45
    alpha1 = torch.tensor(0.0046)  # 0131
    gamma = torch.tensor(0.1777)
    rho = torch.tensor(0.327)
    theta = torch.tensor(2.1070)  # * torch.tensor(.1476)/ kappa
    varphi = torch.tensor(0.068)
    """
    kappa = torch.tensor(0.0553) #0553
    sigma = torch.tensor(0.3325) #3325
    alpha0 = torch.tensor(0.45) #45
    alpha1 = torch.tensor(0.0131) #0131
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542) * torch.tensor(.1476)/ kappa
    varphi = torch.tensor(0.0832)
    """

    # Product specification
    exerciseDates = torch.tensor([5.0, 10.0, 15.0])
    delta = torch.tensor(0.25)
    swapFirstFixingDate = torch.tensor(5.0)
    swapLastFixingDate = torch.tensor(30.0)
    notional = torch.tensor(1e6)

    dTL = torch.linspace(0.0, float(exerciseDates[-1]), 50 * int(exerciseDates[-1]) + 1)

    model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi)

    t = torch.linspace(float(swapFirstFixingDate),
                       float(swapLastFixingDate),
                       int((swapLastFixingDate - swapFirstFixingDate) / delta) + 1)
    strike = model.calc_swap_rate([x.mean(0) for x in model.x0], t, delta)
        #torch.tensor(0.09102013) #model.calc_swap_rate(r0, t, delta)

    rng = RNG(seed=seed, use_av=True)

    # Bermudan Payer Swaption
    bermudan_payer_swpt = BermudanPayerSwaption(
        strike=strike,
        exerciseDates=exerciseDates,
        delta=delta,
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    poly_reg = PolynomialRegressor(deg=deg, use_SVD=True)
    lsmc = LSMC(reg=poly_reg)

    payoff = lsmcDefaultSim(
        prd=bermudan_payer_swpt, mdl=model, rng=rng, N=N, n=n, lsmc=lsmc, reg=poly_reg, dTL=dTL
    )

    price_bermudan_payer_swpt = torch.mean(torch.sum(payoff, dim=0))

    print(f'BermudanPayerSwpt = {price_bermudan_payer_swpt}')

    # European Payer Swaption (lower bound)
    european_payer_swpt = EuropeanPayerSwaption(
        strike=strike,
        exerciseDate=exerciseDates[-1],
        delta=delta,
        swapFirstFixingDate=exerciseDates[-1],
        swapLastFixingDate=swapLastFixingDate,
        notional=notional
    )

    payoff = mcSim(prd=european_payer_swpt, mdl=model, rng=rng, N=N, dTL=dTL)
    price_european_payer_swpt = torch.mean(payoff)

    print(f'EuropeanPayerSwpt (lower bound) = {price_european_payer_swpt}')

    # Sum of European Payer Swaption (upper bound)
    upper_bound = 0.0
    for T in exerciseDates:
        european_payer_swpt = EuropeanPayerSwaption(
            strike=strike,
            exerciseDate=T,
            delta=delta,
            swapFirstFixingDate=T,
            swapLastFixingDate=swapLastFixingDate,
            notional=notional
        )

        payoff = mcSim(prd=european_payer_swpt, mdl=model, rng=rng, N=N, dTL=dTL)
        upper_bound += torch.mean(payoff)

    print(f'EuropeanPayerSwpt (upper bound) = {upper_bound}')

    if plot_lower_bound:
        # European Payer Swaption constituents (lower bound)
        EuPayerSwpts = []
        prds = []
        for e in exerciseDates:
            european_payer_swpt = EuropeanPayerSwaption(
                strike=strike,
                exerciseDate=e,
                delta=delta,
                swapFirstFixingDate=e,
                swapLastFixingDate=swapLastFixingDate,
                notional=notional
            )
            prds.append(european_payer_swpt)

            payoff = mcSim(prd=european_payer_swpt, mdl=model, rng=rng, N=N, dTL=dTL)
            price_european_payer_swpt = torch.mean(payoff)
            EuPayerSwpts.append(price_european_payer_swpt)

        # Create a list of labels for the x-axis
        x_labels = ['{}x{}'.format(int(i), int(swapLastFixingDate - i)) for i in exerciseDates]

        # Create the plot
        plt.figure()
        plt.plot(EuPayerSwpts, '--o', label='European swaption')
        plt.axhline(y=price_bermudan_payer_swpt, linestyle='dashed', color='orange', label='Bermudan swaption')
        plt.xticks(range(len(EuPayerSwpts)), x_labels)  # Set x-axis tick positions and labels
        plt.xlabel('Maturity x Tenor')
        plt.title(
            f'Bermudan payer swaption {int(swapLastFixingDate)}nc{int(exerciseDates[0])} struck at {strike[0].round(decimals=3) * 100}%')
        plt.legend()
        plt.show()

