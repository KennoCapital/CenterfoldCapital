import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd.functional import jvp
from application.engine.vasicek import Vasicek
from application.engine.products import CapletAsPutOnZCB
from application.engine.standard_scalar import DifferentialStandardScaler
from application.engine.differential_Regression import DifferentialPolynomialRegressor
from application.engine.differential_NN import Neural_Approximator
from application.engine.mcBase import mcSim, RNG
from application.utils.path_config import get_plot_path

torch.set_printoptions(4)
torch.set_default_dtype(torch.float64)

if __name__ == '__main__':

    seed = 1234
    N_train = 1024 * 4
    N_test = 256
    use_av = True

    r0_min = -0.02
    r0_max = 0.15

    r0_vec = torch.linspace(r0_min, r0_max, N_train)

    # Differential Neural Network Settings
    seed_weights = 1234
    epochs = 250
    batches_per_epoch = 16
    min_batch_size = 256 * 10
    lam = 1.0
    hidden_units = 20
    hidden_layers = 4

    # Model specification
    a = torch.tensor(0.86)
    b = torch.tensor(0.09)
    sigma = torch.tensor(0.0148)
    measure = 'risk_neutral'

    mdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure=measure)

    rng = RNG(seed=seed, use_av=use_av)


    fig, ax = plt.subplots(2, 2)

    # Product specification
    T = [1.00, 0.50, 0.25, 0.05]
    if len(T) != 4:
        raise ValueError("The length of varying maturities is different to 4!")

    for i in tqdm(range(len(T))):

        exerciseDate = torch.tensor(T[i])
        delta = torch.tensor(0.25)
        swapFirstFixingDate = exerciseDate
        swapLastFixingDate = exerciseDate + torch.tensor(5.0)
        notional = torch.tensor(1e6)

        t_swap_fixings = torch.linspace(
            float(swapFirstFixingDate),
            float(swapLastFixingDate),
            int((swapLastFixingDate - swapFirstFixingDate) / delta + 1)
        )

        strike = torch.tensor(0.0871)

        prd = CapletAsPutOnZCB(
            strike=strike,
            exerciseDate=exerciseDate,
            delta=delta,
            notional=notional
        )

        """ Helper functions for generating training data of pathwise payoffs and deltas """


        def calc_dzcb_dr(r0_vec, t0):
            """
            :param  r0_vec:    Current Short rate r0
            :param  t0:        Current time

            returns:
                tuple with: (Forward Prices, Forward Prices differentiated wrt. r0 evaluated at x)
            """

            def _zcb(r0_vec):
                zcb = mdl.calc_zcb(r0_vec, exerciseDate - t0 + delta)[0]
                return zcb

            ones = torch.ones_like(r0_vec)
            res = jvp(_zcb, r0_vec, ones, create_graph=False)
            return res


        def calc_dcpl_dr(r0_vec, t0):
            """
            :param  r0_vec: Current Short rate r0
            :param  t0:     Current time

            returns:
                tuple with: (Pathwise payoffs, Pathwise differentials wrt. r0 evaluated at x)
            """

            def _payoffs(r0_vec):
                cMdl = Vasicek(a, b, sigma, r0_vec, use_ATS=True, use_euler=False, measure='risk_neutral')
                cPrd = CapletAsPutOnZCB(
                    strike=strike,
                    exerciseDate=exerciseDate - t0,
                    delta=delta,
                    notional=notional
                )
                payoffs = mcSim(cPrd, cMdl, rng, len(r0_vec))
                return payoffs

            ones = torch.ones_like(r0_vec)
            res = jvp(_payoffs, r0_vec, ones, create_graph=False)
            return res

        def training_data(r0_vec: torch.Tensor, t0: float = 0.0, use_av: bool = True):
            if use_av:
                # X_train[i] = X_train[i + N_train],  for all i, when using AV
                r0_vec = torch.concat([r0_vec, r0_vec])

            zcb, dPdr = calc_dzcb_dr(r0_vec, t0)
            y, dydr = calc_dcpl_dr(r0_vec, t0)

            X_train = zcb.reshape(-1, 1)
            y_train = y.reshape(-1, 1)
            z_train = (dydr / dPdr).reshape(-1, 1)

            if use_av:
                idx_half = N_train
                X_train = X_train[:idx_half]
                y_train = 0.5 * (y_train[:idx_half] + y_train[idx_half:])
                z_train = 0.5 * (z_train[:idx_half] + z_train[idx_half:])

            return X_train, y_train, z_train

        """ Calculate `true` swaption price using Monte Carlo for comparison """
        r0_test_vec = torch.linspace(r0_min, r0_max, N_test)
        X_test = mdl.calc_zcb(r0_test_vec, exerciseDate + delta)[0].reshape(-1, 1)
        y_mdl = mdl.calc_cpl(r0_test_vec, exerciseDate, delta, strike, notional)[0].reshape(-1, 1)
        z_mdl = y_mdl.diff(dim=0) / X_test.diff(dim=0)

        """ Estimate Price and Delta using Differential Regression """

        X_train, y_train, z_train = training_data(r0_vec=r0_vec, t0=0.0, use_av=use_av)

        diff_nn = Neural_Approximator(X_train, y_train, z_train)
        diff_nn.prepare(N_train, True, weight_seed=seed_weights, lam=lam, hidden_units=hidden_units,
                        hidden_layers=hidden_layers)
        diff_nn.train(epochs=epochs, batches_per_epoch=batches_per_epoch, min_batch_size=min_batch_size)
        y_pred, z_pred = diff_nn.predict_values_and_derivs(X_test)

        z_mdl /= notional
        z_train /= notional
        z_pred /= notional

        RMSE_price = torch.sqrt(torch.mean((y_pred - y_mdl) ** 2))
        MAE_delta = torch.mean(torch.abs(z_pred[1:] - z_mdl))

        """ Plot results """
        if i < 2:
            row = 0
        else:
            row = 1

        if i % 2 == 0:
            col = 0
        else:
            col = 1

        # Plot delta function
        ax[row, col].plot(X_train, z_train, 'o', color='gray', alpha=0.25, label='Pathwise samples')
        ax[row, col].plot(X_test, z_pred, label='DiffNN', color='orange')
        ax[row, col].plot(X_test[1:], z_mdl, color='black', label='MC (Bump and reval)')
        ax[row, col].set_xlabel('P(0)')
        ax[row, col].set_ylabel('Delta')
        ax[row, col].text(0.05, 0.8, f'MAE = {MAE_delta:.4f}', fontsize=8, transform=ax[row, col].transAxes)

        # Adjust size of subplots
        box0 = ax[row, col].get_position()
        ax[row, col].set_position([box0.x0, box0.y0 - box0.height * 0.1, box0.width, box0.height * 0.9])

        box1 = ax[row, col].get_position()
        ax[row, col].set_position([box1.x0, box1.y0, box1.width, box1.height * 0.9])

        # Title
        ax[row, col].set_title(f'T={T[i]}Y')

    # Legend
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.9))
    fig.suptitle(f'Delta Estimation of 1Y6Y3M European payer swaption using Differential NN, N_train={N_train}', x=0.5, y=0.97)
    plt.show()
    #plt.savefig(get_plot_path('vasicek_AAD_DiffNN_plot_delta_EuPayerSwpt_multiple_snapshots.png'), dpi=400)
