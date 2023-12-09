import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.standard_scalar import DifferentialStandardScaler
from application.experiments.trolle_schwartz.ts_hedge_tools import training_data
from application.engine.products import CapletAsPutOnZCB
from application.engine.trolleSchwartz import trolleSchwartz
from application.utils.path_config import get_plot_path, get_data_path
from application.engine.mcBase import mcSim, RNG
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
    #varphi_min = 0.03
    #varphi_max = 0.13
    #varphi = torch.linspace(varphi_min, varphi_max, N_train)
    # initializer
    varphi_min = 0.03
    varphi_max = 0.13
    varphi = torch.linspace(varphi_min, varphi_max, N_test)

    #varphi = torch.tensor(0.0832)

    model = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)

    rng = RNG(seed=seed, use_av=use_av)

    # Product specification
    exerciseDate = torch.tensor(1.0)
    delta = torch.tensor(0.25)
    notional = torch.tensor(1e6)

    strike = torch.tensor(.07)

    prd = CapletAsPutOnZCB(
        strike=strike,
        exerciseDate=exerciseDate,
        delta=delta,
        notional=notional
    )

    """ Helper functions for generating training data of pathwise payoffs and deltas """


    def calc_dzcb_dv(v0, t0):
        """
        :param  x0_vec:    Possible state variables of state space
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """
        #t0 = torch.tensor(t0)

        #v0_vec = torch.tensor(v0_vec, requires_grad=True)

        v0 = torch.tensor(v0, requires_grad=True)

        def _zcb(v0):
            tmp_mdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            mcSim(prd, tmp_mdl, rng, N_train, dTL)
            state = torch.stack(tmp_mdl.x)[:, :, 0, :]
            zcb = model.calc_zcb(state, t0, exerciseDate + delta)[0]
            return zcb

        zcbs = _zcb(v0) # size N

        J = jacfwd(_zcb, argnums=(0), randomness='same')(v0_vec)

        """
        #jac = jacrev(_zcb, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((model.simDim, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        dzcbs = tmp.unsqueeze(dim=2) # size N x 8 x simDim
        """

        return zcbs, J.flatten()


    def calc_dcpl_dv(v0, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)
        v0 = torch.tensor(v0, requires_grad=True)

        def _payoffs(v0):
            cMdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(varphi), cTL)
            return payoffs

        cpls = _payoffs(v0)

        J = jacfwd(_payoffs, argnums=(0), randomness='same')(v0)

        """

        jac = jacrev(_payoffs, argnums=(0, 1, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((1, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        dcpls = tmp.unsqueeze(dim=2)
        """

        return cpls, J

    def calc_dCdP(zcbs):
        zcbs = torch.tensor(zcbs, requires_grad=True)
        def _payoffs(zcbs,v0=v0):

            varphi = -torch.log(zcbs) / (exerciseDate+delta)

            cMdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(varphi), cTL)
            return payoffs

        cpls = _payoffs(zcbs)
        J = jacfwd(_payoffs, argnums=(0), randomness='same')(zcbs)

        return cpls, J.sum_to_size(len(zcbs))


    def calc_dXdv(v0):

        def _x(v0):
            tmp_mdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
            mcSim(prd, tmp_mdl, rng, N_train, dTL)
            x, v, phi1, phi2, phi3, phi4, phi5, phi6 = tmp_mdl.x

            # torch.stack(tmp_mdl.x)[:, :, -1, :]
            return (x, v, phi1, phi2, phi3, phi4, phi5, phi6)

        v0 = torch.tensor(v0, requires_grad=True)
        dvecs = jacfwd(_x, has_aux=False, randomness='same')(v0)  # list of state vars sens. to v0
        jac = torch.stack(dvecs)

        return None


    hedge_points = 50
    dTL = torch.linspace(0.0, exerciseDate, int(hedge_points * exerciseDate + 1))

    mcSim(prd, model, rng, N_train, dTL)
    x0_vec = torch.stack(model.x)[:, :, 0, :]

    """ Calculate  sens """
    t0 = torch.tensor(0.)
    v0_vec = torch.tensor([v0*0.5, v0, v0*2])
    C, dC = [], []

    for v in tqdm(v0_vec):
        cpl, dcpl = calc_dcpl_dv(v0, t0)
        C.append(cpl.detach())
        dC.append(dcpl.detach())

    zcbs = torch.exp(-varphi*(exerciseDate+delta))

    plt.figure()
    plt.plot(zcbs, dC[0]/1000,'o' )
    plt.plot(zcbs, dC[1]/1000, 'o')
    plt.plot(zcbs, dC[2]/1000, 'o')
    plt.plot()

    v0_vec = torch.linspace(0.01, 0.1, N_test)
    x_test = torch.exp(-varphi*(exerciseDate+delta))
    y_mdl = torch.full_like(v0_vec, torch.nan)


    for j in tqdm(range(len(v0_vec))):
        tmp_mdl = trolleSchwartz(v0_vec[j], gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi[j], simDim=1)
        tmp_rng = RNG(seed=seed, use_av=use_av)
        y_mdl[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 10000, dTL)))
    y_mdl = y_mdl.reshape(-1, 1)
    z_mdl = y_mdl.diff(dim=0) / v0_vec.diff(dim=0)

    plt.figure()
    plt.plot(x_test[1:], z_mdl, 'o')
    plt.xlabel(r'$v0$')
    plt.ylabel(r'$dC / dv_0 $')
    plt.show()

    """ Calculate `true` caplet price using Monte Carlo for comparison """
    #X_test = torch.exp(-varphi * exerciseDate).reshape(-1, 1)



    """
    # Required when using AV to concat the initial forward rates
    if use_av:
        varphi = torch.concat([varphi, varphi], dim=0)
        model.varphi = varphi

    

    X_train, y_train, z_train = training_data(x0_vec=x0_vec,
                                              t0=0.0,
                                              calc_dPrd_dr=calc_dcpl_dx,
                                              calc_dU_dr=calc_dzcb_dx,
                                              use_av=use_av)

    plt.figure()
    plt.plot(X_train, y_train, 'o')
    plt.show()

    v0_vec = torch.linspace(0.01, 0.1, N_test)
    y_mdl = torch.full_like(v0_vec, torch.nan)

    for j in tqdm(range(len(v0_vec))):
        tmp_mdl = trolleSchwartz(v0_vec[j], gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
        tmp_rng = RNG(seed=seed, use_av=use_av)
        y_mdl[j] = (torch.mean(mcSim(prd, tmp_mdl, tmp_rng, 10000, dTL)))
    y_mdl = y_mdl.reshape(-1, 1)
    z_mdl = y_mdl.diff(dim=0) / v0_vec.diff(dim=0)

    plt.figure()
    plt.plot(v0_vec, z_mdl,  'o')
    plt.xlabel(r'$v0$')
    plt.ylabel(r'$dC / dv_0 $')
    plt.show()

    t0 = torch.tensor(0.)
    p, dp = calc_dzcb_dx(x0_vec, t0)
    c, dc = calc_dcpl_dx(x0_vec, t0)

    plt.figure()
    plt.plot(p, dc[:, 1, 0], 'o')
    plt.ylabel('dc/dv')
    plt.xlabel('P')
    plt.show()
    
    
    
    
    plt.figure()
    plt.plot(P[0].detach().numpy(), dP[0].detach().numpy(), 'o')
    plt.plot(P[-1].detach().numpy(), dP[-1].detach().numpy(), 'o')
    plt.show()


    c2, dcdp = calc_dCdP(P[-1])

    plt.figure()
    plt.plot(P[-1].detach().numpy(), dcdp.detach().numpy()*dP[-1].detach().numpy(), 'o')
    plt.show()
    """