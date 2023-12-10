import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.experiments.trolle_schwartz.ts_hedge_tools import training_data
from application.engine.products import CapletAsPutOnZCB
from application.engine.trolleSchwartz import trolleSchwartz
from application.utils.path_config import get_plot_path, get_data_path
from application.engine.mcBase import mcSim, RNG, mcSimPaths
from torch.func import jacrev, jacfwd
import pickle
import os

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024
    N_test = 256
    use_av = True
    save_plot = False

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True)
    scalar = DifferentialStandardScaler()

    # Trolle-Schwartz model specification
    kappa = torch.tensor(0.0553)
    sigma = torch.tensor(0.3325)
    alpha0 = torch.tensor(0.045)
    alpha1 = torch.tensor(0.131)
    gamma = torch.tensor(0.3341)
    rho = torch.tensor(0.4615)
    theta = torch.tensor(0.7542) * kappa / torch.tensor(2.1476)
    v0 = theta

    # initializer
    varphi_min = 0.03
    varphi_max = 0.13
    varphi = torch.linspace(varphi_min, varphi_max, N_train)

    model = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = torch.tensor(.07)

    prd1 = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    prd2 = CapletAsPutOnZCB(
        strike=strike*1.25,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    hedge_points = 100
    # dTL = torch.linspace(0.0, float(exerciseDate+delta), hedge_points + 1)
    dTL = torch.linspace(0.0, exerciseDate, int(hedge_points * exerciseDate + 1))

    cf = torch.zeros((2,len(dTL),N_train))
    P = torch.zeros((len(dTL),N_train))

    for i, t in enumerate(dTL):
        cf1 = mcSim(prd1, model, rng, N_train, dTL[i:-1])

        torch.stack(model.x)
        P[i,:] = model.calc_zcb(torch.stack(model.x), t, exerciseDate+ delta)

        cf2 = mcSim(prd2, model, rng, N_train, dTL[i:-1])

        cf[0, i, :] = cf1
        cf[1, i, :] = cf2


    plt.figure()
    plt.plot(dTL,  (cf[0,:,:] - cf[1,:,:]).std(dim=1), 'o')
    plt.show()


