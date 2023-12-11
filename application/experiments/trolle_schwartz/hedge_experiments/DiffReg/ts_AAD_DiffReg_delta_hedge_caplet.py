import torch
import matplotlib.pyplot as plt
from torch.autograd.functional import jvp
from tqdm import tqdm
from application.engine.vasicek import Vasicek, choose_training_grid
from application.engine.products import Caplet, CapletAsPutOnZCB
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.mcBase import mcSimPaths, mcSim, RNG
from application.utils.path_config import get_plot_path
from application.utils.torch_utils import max0
from application.experiments.trolle_schwartz.ts_hedge_tools import diff_reg_fit_predict, training_data
from application.experiments.trolle_schwartz.ts_hedge_tools import calc_delta_diff_nn
from application.engine.trolleSchwartz import trolleSchwartz
import torch
from application.engine.mcBase import mcSim, RNG

from torch.func import jacrev
torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 256
    N_test = 256
    use_av = True

    save_plot = False

    hedge_points = 20

    # Setup Differential Regressor, and Scalar
    deg = 9
    alpha = 1.0
    diff_reg = DifferentialPolynomialRegressor(deg=deg, alpha=alpha, use_SVD=True, bias=True, include_interactions=True)


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
    #varphi = torch.linspace(varphi_min, varphi_max, N_train*(1+use_av))
    #varphi = torch.ones(N_train) * 0.0832
    varphi = torch.tensor(0.0832)

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

    # Simulate paths
    dTL = torch.linspace(0.0, float(exerciseDate), hedge_points + 1)
    cashflows = mcSim(prd, model, rng, N_test, dTL)

    x = torch.stack(model.x)
    x0_vec = x[:, :, 0, :]


    # Find index of the exercise date
    last_idx = int((dTL == exerciseDate).nonzero(as_tuple=True)[0])

    """ Helper functions for calculating pathwise payoffs and deltas, and generating training data """


    def calc_dzcb_dx(x0_vec, t0):
        """
        :param  x0_vec:    Possible state variables of state space
        :param  t0:        Current time

        returns:
            tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)

        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [x.reshape(model.simDim, -1) for x in x0_vec]

        def _zcb(x, v, phi1, phi2, phi3, phi4, phi5, phi6):
            state = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
            zcb = model.calc_zcb(state, t0, exerciseDate + delta)[0]
            return zcb

        zcbs = _zcb(x, v, phi1, phi2, phi3, phi4, phi5, phi6)  # size N

        # first we find x, phis state var sens (since the method 'calc_zcb' will not give us any sens wrt v)
        # the reverse mode is efficient for R^n -> R^m when n>m
        jac = jacrev(_zcb, argnums=(0, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((model.simDim, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        tmp = tmp.unsqueeze(dim=2)  # size N x 7 (!) x simDim

        return zcbs[0], tmp


    def calc_dcpl_dx(x0_vec, t0):
        """
        :param  r0_vec: Current Short rate r0
        :param  t0:     Current time

        returns:
            tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
        """
        t0 = torch.tensor(t0)
        x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [x.reshape(model.simDim, -1) for x in x0_vec]

        def _payoffs(x, v, phi1, phi2, phi3, phi4, phi5, phi6):
            state = [x, v, phi1, phi2, phi3, phi4, phi5, phi6]
            f0T = model.calc_instant_fwd(state, t0, exerciseDate + delta - t0).flatten()
            cMdl = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, f0T, simDim=1)
            cPrd = CapletAsPutOnZCB(
                strike=strike,
                exerciseDate=exerciseDate - t0,
                delta=delta,
                notional=notional
            )
            cTL = dTL[dTL <= exerciseDate - t0]
            payoffs = mcSim(cPrd, cMdl, rng, len(f0T), cTL)
            return payoffs

        cpls = _payoffs(x, v, phi1, phi2, phi3, phi4, phi5, phi6)

        # the reverse mode is efficient for R^n -> R^m when n>m
        jac = jacrev(_payoffs, argnums=(0, 2, 3, 4, 5, 6, 7))(x, v, phi1, phi2, phi3, phi4, phi5, phi6)
        jac_sum = torch.stack([x.sum_to_size((1, x0_vec.shape[2])) for x in jac])
        tmp = jac_sum.permute(1, 2, 0).squeeze()
        tmp = tmp.unsqueeze(dim=2)  # size N x 7 (!) x simDim

        return cpls, tmp



    """ Delta Hedge Experiment """
    t0 = torch.tensor(0.)

    # Initialize experiment
    B = torch.ones((N_test,))
    zcb = model.calc_zcb(x0_vec, t0, exerciseDate + delta).flatten()

    # Get price of claim (we use 500k simulations to get an accurate estimate)
    cplMC = torch.empty_like(B)
    for i in tqdm(range(N_test), desc='Estimating initial value of Caplet with MC'):
        mdl_pricer = trolleSchwartz(v0, gamma, kappa, theta, rho, sigma, alpha0, alpha1, varphi, simDim=1)
        TL = torch.linspace(0.0, float(exerciseDate), 100 + 1)
        cplMC[i] = torch.mean(mcSim(prd, mdl_pricer, rng, 10000, TL))

    V = cplMC
    h_a = diff_reg_fit_predict(u_vec=zcb, t0=0.0, x0_vec=x0_vec,
                               calc_dU_dr=calc_dzcb_dx, calc_dPrd_dr=calc_dcpl_dx,
                               diff_reg=diff_reg, use_av=use_av)[1].flatten()
    h_b = (V - h_a * zcb) / B
    V_values = [V]

    # Loop over time
    for k in tqdm(range(1, len(dTL)), desc='Running Delta hedge'):
        dt = dTL[k] - dTL[k - 1]
        t = dTL[k]

        # Update market variables
        zcb = model.calc_zcb(x[:,:, k, :], t0, exerciseDate - t + delta).flatten()

        # Update portfolio
        r = model.calc_short_rate(x[:, :, k, :], t)
        rlag = model.calc_short_rate(x[:, :, k-1, :], dTL[k-1])
        B *= torch.exp(0.5 * (r + rlag) * dt).flatten()
        V = h_a * zcb + h_b * B

        V_values.append(V)

        if k < len(dTL) - 1:
            x0_vec = x[:, :, k, :]#choose_training_grid(x[:, :, k, :], N_train)
            h_a = diff_reg_fit_predict(u_vec=zcb, x0_vec=x0_vec, t0=t,
                                       calc_dU_dr=calc_dzcb_dx, calc_dPrd_dr=calc_dcpl_dx,
                                       diff_reg=diff_reg, use_av=use_av)[1].flatten()
            h_b = (V - h_a * zcb) / B

    V_values = torch.vstack(V_values)

    zcbT = model.calc_zcb(x[:, :, -1, :].sort().values, t0, delta).flatten()
    df = model.calc_zcb(x[:, :, -1, :].sort().values, t0, delta).flatten()
    K_bar = 1.0 + delta * strike
    payoff_func = notional * K_bar * max0(1.0 / K_bar - zcbT) * df

    V *= model.calc_zcb(x[:, :, -1, :], t0, delta).flatten()

    MAE_value = torch.mean(torch.abs(V - notional * K_bar * max0(1.0 / K_bar - model.calc_zcb(x[:, :, -1, :], t0, delta).flatten())))

    """ Plot """
    av_str = 'with AV' if use_av else 'without AV'

    fig, ax = plt.subplots(1)
    ax.plot(zcbT, payoff_func, color='black', label='Payoff function')
    ax.plot(model.calc_zcb(x[:, :, -1, :], t0, delta).flatten(), V, 'o', color='orange', label='Value of Hedge Portfolio', alpha=0.5)
    ax.set_xlabel('ZCB(T)')
    ax.text(0.05, 0.8, f'MAE = {MAE_value:,.2f}', fontsize=8, transform=ax.transAxes)

    # Adjust size of plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])

    # Title
    fig.suptitle(
        prd.name + f'\nHedgeFreq={dTL[1]:.4g}, alpha = {alpha}, deg={deg}, {N_train} training samples ' + av_str)

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=2, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.90))
    if save_plot:
        plt.savefig(get_plot_path('ts_AAD_DiffReg_caplet_as_put_on_zcb_delta_hedge.png'), dpi=400)
    plt.show()
