from application.engine.AAD import computeJacobian_dCdr, computeJacobian_dFdr
from application.engine.mcBase import mcSim, RNG
from application.engine.products import Caplet
from application.engine.vasicek import Vasicek
from application.engine.differential_regression import DifferentialRegression, diffreg_fit
from application.utils.torch_utils import max0
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import torch
import numpy as np

def find_r_for_target_fwd(target_fwd_rate, model, start, delta):
    def objective(r):
        return model.calc_fwd(r, start, delta) - target_fwd_rate

    result = root_scalar(objective, bracket=[-1, 1], method='brentq', xtol=1e-15)
    return result.root


if __name__ == '__main__':
    torch.set_printoptions(8)
    torch.set_default_dtype(torch.float64)

    N = 1000
    train_sz = N
    test_sz = 200

    degree = 5
    alpha = 1.0

    measure = 'risk_neutral'

    show_r = False # running a single prediction of price and delta for the short rate sensititivity
    show_base = False # running a single prediction of price and delta of the underlying forward rate
    delta_hedge_bump = False # conduct delta hedge using bump and revalue delta
    delta_hedge_dML = False # conduct delta hedge using differential regression
    delta_convergence = True # conduct delta hedge using differential regression
                              # for multiple hedge points to illustrate hedge error


    a = torch.tensor(0.86)
    b = torch.tensor(0.08)
    sigma = torch.tensor(0.0148)
    r0 = torch.tensor(0.08)
    bump = 0.0001 # for delta computing
    Notional = 40

    start = torch.tensor(4.75)
    delta = torch.tensor(0.25)
    expiry = start + delta

    dTL = torch.linspace(0.0, float(start), 20)  # discretization TL

    t = torch.linspace(float(start), float(expiry), 2)  # int(expiry/delta))

    model = Vasicek(a, b, sigma, r0, use_ATS=False, measure=measure)
    swap_rate = model.calc_swap_rate(r0, t, delta)
    prd = Caplet(
        strike=swap_rate,
        start=start,
        delta=delta
    )

    if delta_convergence:
        print("running delta convergence")
        hedge_errors = []
        hedge_errors_avg = []
        hedge_points = [10, 20, 30, 40, 50]
        bank_book = []

        for h in hedge_points:
            print("hedge point: ", h)
            dTL = torch.linspace(0.0, float(start), h)

            seed = 1234
            rng = RNG(seed=seed, use_av=True)

            price = mcSim(prd, model, rng, N, dTL)
            r = model.x
            V = model.calc_cpl(r0, start, prd.delta, swap_rate).detach().numpy().reshape(-1) * Notional
            fwd = model.calc_fwd(r[0, :], prd.start, prd.delta).detach().numpy().reshape(-1, 1)

            spot_grid = torch.linspace(r0.detach().numpy() - 0.03, r0.detach().numpy() + 0.03, N)
            diffreg_mdl, x, y, z = diffreg_fit(prd=prd,
                                               mdl=model,
                                               rng=rng,
                                               s=dTL[0],
                                               rs=spot_grid,
                                               measure='terminal')
    
            _, delta_F = diffreg_mdl.predict(fwd, predict_derivs=True)
            delta_F = delta_F.reshape(-1)
            fwd = fwd.reshape(-1)

            # TODO: Delete if do not want to consider hedging with bump and revalue delta
            #target_fwd_rates = (fwd + bump).reshape(-1)
            #required_rs = torch.tensor([find_r_for_target_fwd(rate, model, start, delta) for rate in target_fwd_rates])
            #fwd_bump = model.calc_fwd(required_rs, start, delta).detach().numpy().reshape(-1)
            #delta_F = (model.calc_cpl(required_rs, start, prd.delta, swap_rate).detach().numpy() - V) / (fwd_bump - fwd)

            b = (V - delta_F * fwd * Notional)
            dt = start.detach().numpy() / (dTL.numel() - 1)

            for i in range(1, dTL.numel() - 1):
                print(f"iterating over hedgepoint {i}:")
                print(f"We are at timepoint {dTL[i]}")
                fwd = model.calc_fwd(r[i, :], start - dTL[i], delta).detach().numpy().reshape(-1, 1)
                DF = torch.exp((r[i, :] - r[i-1,:]) / 2 * dt).detach().numpy().reshape(-1)
                V = Notional * delta_F * fwd.reshape(-1) + b * DF

                spot_grid = torch.linspace(float(r[i, :].min() - r[i, :].std()), float(r[i, :].max() + r[i, :].std()), N)
                diffreg_mdl, x, y, z = diffreg_fit(prd=prd,
                                                   mdl=model,
                                                   rng=rng,
                                                   s=dTL[i],
                                                   rs=spot_grid,
                                                   measure='terminal')
                _, delta_F = diffreg_mdl.predict(fwd, predict_derivs=True)
                delta_F = delta_F.reshape(-1)
                fwd = fwd.reshape(-1)

                # TODO: Delete if do not want to consider hedging with bump and revalue delta
                #cpl_val = model.calc_cpl(r[i, :], start - dTL[i - 1], prd.delta, swap_rate).detach().numpy()
                #target_fwd_rates = (fwd + bump).reshape(-1)
                #required_rs = torch.tensor([find_r_for_target_fwd(rate, model, start - dTL[i - 1], delta) for rate in target_fwd_rates])
                #fwd_bump = model.calc_fwd(required_rs, start - dTL[i - 1], delta).detach().numpy().reshape(-1)
                #cpl_val_bump = model.calc_cpl(required_rs, start - dTL[i - 1], prd.delta, swap_rate).detach().numpy()
                #delta_F = (cpl_val_bump - cpl_val) / (fwd_bump - fwd)

                b = V - delta_F * fwd * Notional

            fwd = model.calc_fwd(r[-1, :], start - dTL[-1], delta).detach().numpy().reshape(-1)
            DF = torch.exp(r[-1, :] * dt).detach().numpy().reshape(-1)
            V = Notional * delta_F * fwd + b * DF
            option_payoff = Notional * delta * max0(torch.tensor(fwd) - swap_rate)
            hedge_error = torch.tensor(V) - option_payoff

            sorted_indices = np.argsort(fwd)
            sorted_fwd = fwd[sorted_indices]
            sorted_payoff = option_payoff[sorted_indices]

            if h == hedge_points[-1]:
                plt.figure(figsize=(10, 6))
                plt.plot(sorted_fwd, sorted_payoff, linewidth=5, color='gray', label='Payoff')
                plt.scatter(fwd, V, marker='o', c='red', label='Value of Portfolio', s=50, alpha=0.2, zorder=5)
                plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
                plt.title(f'Option Payoff vs Portfolio Value - Differential Regression h={h}', fontsize=16, fontweight='bold')
                plt.xlabel('Forward Rate', fontsize=14)
                plt.ylabel('Value', fontsize=14)
                plt.legend(loc='upper left', fontsize=12)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.show()

            std_hedge_error = torch.std(hedge_error).item()
            hedge_errors.append(std_hedge_error)

            avg_hedge_error = torch.mean(hedge_error).item()
            hedge_errors_avg.append(avg_hedge_error)

            bank_book.append(b.mean())

        plt.figure()
        plt.title('Average Hedge Errors')
        plt.plot(hedge_points, hedge_errors_avg, 'o', label='Hedge Errors')
        plt.xlabel('Hedge Points')
        plt.ylabel('Avg. of hedge errors')
        plt.gca().set_xticks(np.array(hedge_points))
        plt.yticks(fontsize=12)
        plt.show()

        plt.figure()
        plt.title('Average amount kept in the bank account')
        plt.plot(hedge_points, bank_book, 'o', label='Hedge Errors')
        plt.xlabel('Hedge Points')
        plt.ylabel('Avg. of bank account kept')
        plt.gca().set_xticks(np.array(hedge_points))
        plt.yticks(fontsize=12)
        plt.show()



        x_vals = np.array(hedge_points)
        y_vals = x_vals ** -0.5  # Convergence order of 0.5
        plt.figure()
        plt.title('Convergence Order of Hedge Errors')
        #plt.loglog(x_vals, y_vals * min(hedge_errors), '--', label='Convergence Order 0.5')  # Scale to make it visible
        plt.loglog(hedge_points, hedge_errors, '-o', label='Hedge Errors')
        plt.xlabel('Hedge Points')
        plt.ylabel('Std. of Hedge Errors')
        plt.gca().set_xticks([], minor=True)
        plt.gca().set_xticks(x_vals)
        plt.gca().set_xticklabels(x_vals.astype(str))
        plt.gca().set_yticks([], minor=True)
        plt.gca().set_yticks(np.array(hedge_errors))
        plt.gca().set_yticklabels(np.array(hedge_errors).round(4).astype(str))
        plt.show()




    if delta_hedge_dML:
        print("running delta hedge with differential regression")
        seed = 1234
        rng = RNG(seed=seed, use_av=True)

        price = mcSim(prd, model, rng, N, dTL)
        r = model.x
        V = model.calc_cpl(r0, start, prd.delta, swap_rate).detach().numpy().reshape(-1) * Notional
        fwd = model.calc_fwd(r[0, :], prd.start, prd.delta).detach().numpy().reshape(-1, 1)

        spot_grid = torch.linspace(r0.detach().numpy() - 0.03, r0.detach().numpy() + 0.03, N)
        diffreg_mdl, x, y, z = diffreg_fit(prd=prd,
                                           mdl=model,
                                           rng=rng,
                                           s=dTL[0],
                                           rs=spot_grid,
                                           measure='terminal',
                                           dtl=dTL)

        _, delta_F = diffreg_mdl.predict(fwd, predict_derivs=True)
        delta_F = delta_F.reshape(-1)


        fwd = fwd.reshape(-1)

        b = (V - delta_F * fwd * Notional)
        dt = start.detach().numpy() / (dTL.numel() - 1)

        for i in range(1, dTL.numel() - 1):
            print(f"iterating over hedgepoint {i}:")
            print(f"We are at timepoint {dTL[i]}")
            fwd = model.calc_fwd(r[i, :], start - dTL[i], delta).detach().numpy().reshape(-1, 1)
            DF = torch.exp((r[i, :] + r[i - 1, :]) / 2 * dt).detach().numpy().reshape(-1)
            V = Notional * delta_F * fwd.reshape(-1) + b * DF

            spot_grid = torch.linspace(float(r[i, :].min() - r[i, :].std()), float(r[i, :].max() + r[i, :].std()), N)
            diffreg_mdl, x, y, z = diffreg_fit(prd=prd,
                                               mdl=model,
                                               rng=rng,
                                               s=dTL[i],
                                               rs=spot_grid,
                                               measure='terminal',
                                               dtl=dTL[i:])

            _, delta_F = diffreg_mdl.predict(fwd, predict_derivs=True)
            delta_F = delta_F.reshape(-1)

            fwd = fwd.reshape(-1)

            b = V - delta_F * fwd * Notional

        fwd = model.calc_fwd(r[-1, :], start - dTL[-1], delta).detach().numpy().reshape(-1)
        DF = torch.exp(r[-1, :] * dt).detach().numpy().reshape(-1)
        V = Notional * delta_F * fwd + b * DF
        option_payoff = Notional * delta * max0(torch.tensor(fwd) - swap_rate)
        hedge_error = (torch.tensor(V) - option_payoff)

        sorted_indices = np.argsort(fwd)
        sorted_fwd = fwd[sorted_indices]
        sorted_payoff = option_payoff[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_fwd, sorted_payoff, linewidth=5, color='gray', label='Payoff')
        plt.scatter(fwd, V, marker='o', c='red', label='Value of Portfolio', s=50, alpha=0.2, zorder=5)
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.title('Option Payoff vs Portfolio Value - Differential Regression', fontsize=16, fontweight='bold')
        plt.xlabel('Forward Rate', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(loc='upper left', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()






    if delta_hedge_bump:
        print("running delta hedge with bump and revalue")
        seed = 1234
        rng = RNG(seed=seed, use_av=True)

        price = mcSim(prd, model, rng, N, dTL)
        r = model.x
        V = model.calc_cpl(r0, start, prd.delta, swap_rate).detach().numpy().reshape(-1)
        fwd = model.calc_fwd(r[0, :], prd.start, prd.delta).detach().numpy().reshape(-1, 1)

        fwd = fwd.reshape(-1)
        target_fwd_rates = (fwd + bump).reshape(-1)
        required_rs = torch.tensor([find_r_for_target_fwd(rate, model, start, delta) for rate in target_fwd_rates])
        fwd_bump = model.calc_fwd(required_rs, start, delta).detach().numpy().reshape(-1)
        delta_F = (model.calc_cpl(required_rs, start, delta, swap_rate).detach().numpy().reshape(-1) - V) / (fwd_bump - fwd)


        b = (V - delta_F * fwd)
        dt = start.detach().numpy() / (dTL.numel() - 1)

        for i in range(1, dTL.numel() - 1):
            print(f"iterating over hedgepoint {i}:")
            print(f"We are at timepoint {dTL[i]}")
            fwd = model.calc_fwd(r[i, :], start - dTL[i], delta).detach().numpy().reshape(-1, 1)
            DF = torch.exp((r[i, :] + r[i-1,:])/2 * dt).detach().numpy().reshape(-1)
            V = delta_F * fwd.reshape(-1) + b * DF

            fwd = fwd.reshape(-1)

            cpl_val = model.calc_cpl(r[i, :], start - dTL[i], delta, swap_rate).detach().numpy().reshape(-1)

            target_fwd_rates = (fwd + bump).reshape(-1)
            required_rs = torch.tensor([find_r_for_target_fwd(rate, model, start - dTL[i], delta) for rate in target_fwd_rates])
            fwd_bump = model.calc_fwd(required_rs, start - dTL[i], delta).detach().numpy().reshape(-1)
            cpl_val_bump = model.calc_cpl(required_rs, start - dTL[i], prd.delta, swap_rate).detach().numpy().reshape(-1)
            delta_F = (cpl_val_bump - cpl_val) / (fwd_bump - fwd)

            b = V - delta_F * fwd

            option_payoff = delta * max0(torch.tensor(fwd) - swap_rate)
            hedge_error = torch.tensor(V) - option_payoff




        fwd = model.calc_fwd(r[-1, :], start - dTL[-1], delta).detach().numpy().reshape(-1)
        DF = torch.exp(r[-1, :] * dt).detach().numpy().reshape(-1)
        V = delta_F * fwd + b * DF
        option_payoff = delta * max0(torch.tensor(fwd) - swap_rate)
        hedge_error = torch.tensor(V) - option_payoff

        sorted_indices = np.argsort(fwd)
        sorted_fwd = fwd[sorted_indices]
        sorted_payoff = option_payoff[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_fwd, sorted_payoff, linewidth=5, color='gray', label='Payoff')
        plt.scatter(fwd, V, marker='o', c='red', label='Value of Portfolio', s=50, alpha=0.2, zorder=5)
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.title('Option Payoff vs Portfolio Value', fontsize=16, fontweight='bold')
        plt.xlabel('Forward Rate', fontsize=14)
        plt.ylabel('Value', fontsize=14)
        plt.legend(loc='upper left', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()


    if show_base:
        seed = [1234, 5678]

        model.sigma = sigma * 2.0
        model.allocate(prd, N, dTL)

        rng = RNG(seed=seed[0], use_av=True)
        rng.N = N
        rng.M = len(model.timeline) - 1

        # Draw random variables
        Z = rng.gaussMat()

        # simulate spot grid
        model.simulate(Z)
        spot_grid = model.x[-1,:]

        # set spot grid as r0 on the model:
        spot_grid.requires_grad_()
        model.r0 = spot_grid
        model.sigma = sigma

        rng = RNG(seed=seed[1], use_av=True)

        y_train = mcSim(prd, model, rng, N, dTL)
        r = model.x

        r_test = r[-1, :]
        upper_mask = r_test <= spot_grid.max() - r_test.std()
        r_test = torch.masked_select(r_test, upper_mask)
        lower_mask = r_test >= spot_grid.min() + r_test.std()
        r_test = torch.masked_select(r_test, lower_mask)


        x_train = model.calc_fwd(spot_grid, start, delta)


        dCdr = torch.sum(computeJacobian_dCdr(prd, model, rng, N, spot_grid, dTL), dim=1)
        dFdr = torch.sum(computeJacobian_dFdr(model, spot_grid, start, delta), dim=1)
        z_train = dCdr / dFdr

        x_train = x_train.detach().numpy().reshape(-1, 1)
        y_train = y_train.detach().numpy().reshape(-1, 1)
        z_train = z_train.detach().numpy().reshape(-1, 1)


        y_train_mdl_cpl = model.calc_cpl(spot_grid, start, delta, swap_rate).reshape(-1)

        target_fwd_rates = (x_train + bump).reshape(-1)
        required_rs = torch.tensor([find_r_for_target_fwd(rate, model, start, delta) for rate in target_fwd_rates])
        fwd_bump = model.calc_fwd(required_rs, start, delta).detach().numpy().reshape(-1)
        bump_fwd = fwd_bump - x_train.reshape(-1)

        y_train_mdl_cpl_bump = model.calc_cpl(required_rs, start, delta, swap_rate).reshape(-1)

        y_train_mdl_cpl = y_train_mdl_cpl.detach().numpy()
        y_train_mdl_cpl_bump = y_train_mdl_cpl_bump.detach().numpy()

        z_train_mdl_cpl = (y_train_mdl_cpl_bump - y_train_mdl_cpl) / bump_fwd #bump

        x_test = model.calc_fwd(r_test, start, delta).detach().numpy().reshape(-1, 1)

        y_test_mdl_cpl = model.calc_cpl(r_test, start, delta, swap_rate).reshape(-1)


        target_fwd_rates = (x_test + bump).reshape(-1)
        required_rs = torch.tensor([find_r_for_target_fwd(rate, model, start, delta) for rate in target_fwd_rates])
        fwd_bump = model.calc_fwd(required_rs, start, delta).detach().numpy().reshape(-1)
        bump_fwd = fwd_bump - x_test.reshape(-1)

        y_test_mdl_cpl_bump = model.calc_cpl(required_rs, start, delta, swap_rate).reshape(-1)

        y_test_mdl_cpl = y_test_mdl_cpl.detach().numpy()

        y_test_mdl_cpl_bump = y_test_mdl_cpl_bump.detach().numpy()

        z_test_mdl_cpl = (y_test_mdl_cpl_bump - y_test_mdl_cpl) / bump_fwd #bump

        diffreg = DifferentialRegression(degree=degree, alpha=alpha)
        diffreg.fit(x_train, y_train, z_train)


        y_pred, z_pred = diffreg.predict(x_test, predict_derivs=True)

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.title(f"price training plot for caplet with expiry {start}")
        plt.plot(x_train, y_train, 'o', label='train')
        plt.plot(x_train, y_train_mdl_cpl, 'o', label='black')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.title(f"price testing plot for caplet with expiry {start}")
        plt.plot(x_test, y_pred, 'o', label='predictions')
        plt.plot(x_test, y_test_mdl_cpl, 'o', label='black')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.title(f"delta training plot for caplet with expiry {start}")
        plt.plot(x_train, z_train, 'o', label='train')
        plt.plot(x_train, z_train_mdl_cpl, 'o', label='bump and reval')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.title(f"delta testing plot for caplet with expiry {start}")
        plt.plot(x_test, z_pred, 'o', label='predictions')
        plt.plot(x_test, z_test_mdl_cpl, 'o', label='bump and reval')
        plt.legend()

        plt.tight_layout()
        plt.show()
































    if show_r:
        seed = [1234, 5678]

        model.sigma = sigma * 2.0
        model.allocate(prd, N, dTL)

        rng = RNG(seed=seed[0], use_av=True)
        rng.N = N
        rng.M = len(model.timeline) - 1
        # Draw random variables
        Z = rng.gaussMat()
        # simulate spot grid
        model.simulate(Z)
        spot_grid = model.x[-1, :]

        # set spot grid as r0 on the model:
        spot_grid.requires_grad_()
        model.r0 = spot_grid
        model.sigma = sigma

        rng = RNG(seed=seed[1], use_av=True)

        y_train = mcSim(prd, model, rng, N, dTL)
        r = model.x

        r_test = r[-1, :]
        upper_mask = r_test <= spot_grid.max() - r_test.std()
        r_test = torch.masked_select(r_test, upper_mask)
        lower_mask = r_test >= spot_grid.min() + r_test.std()
        r_test = torch.masked_select(r_test, lower_mask)

        # x_train = model.calc_fwd(spot_grid, start, delta)
        x_train = spot_grid

        dCdr = torch.sum(computeJacobian_dCdr(prd, model, rng, N, spot_grid, dTL), dim=1)
        dFdr = torch.sum(computeJacobian_dFdr(model, spot_grid, start, delta), dim=1)
        # z_train = dCdr / dFdr
        z_train = dCdr

        x_train = x_train.detach().numpy().reshape(-1, 1)
        y_train = y_train.detach().numpy().reshape(-1, 1)
        z_train = z_train.detach().numpy().reshape(-1, 1)

        x_train_fwd = model.calc_fwd(spot_grid, start, delta).detach().numpy()
        x_train_fwd_bump = model.calc_fwd(spot_grid + bump, start, delta).detach().numpy()
        dFdr_bump = (x_train_fwd_bump - x_train_fwd) / bump

        dFdr = dFdr.detach().numpy()

        y_train_mdl_cpl = model.calc_cpl(spot_grid, start, delta, swap_rate).reshape(-1)
        y_train_mdl_cpl = y_train_mdl_cpl.detach().numpy()

        y_train_mdl_cpl_bump = model.calc_cpl(spot_grid + bump, start, delta, swap_rate).reshape(-1)
        y_train_mdl_cpl_bump = y_train_mdl_cpl_bump.detach().numpy()

        z_train_mdl_cpl = (y_train_mdl_cpl_bump - y_train_mdl_cpl) / bump

        # x_test = model.calc_fwd(r_test, start, delta).detach().numpy().reshape(-1, 1)
        x_test = r_test.detach().numpy().reshape(-1, 1)

        y_test_mdl_cpl = model.calc_cpl(r_test, start, delta, swap_rate).reshape(-1)
        y_test_mdl_cpl = y_test_mdl_cpl.detach().numpy()

        y_test_mdl_cpl_bump = model.calc_cpl(r_test + bump, start, delta, swap_rate).reshape(-1)
        y_test_mdl_cpl_bump = y_test_mdl_cpl_bump.detach().numpy()

        z_test_mdl_cpl = (y_test_mdl_cpl_bump - y_test_mdl_cpl) / bump

        diffreg = DifferentialRegression(degree=degree, alpha=alpha)
        diffreg.fit(x_train, y_train, z_train)


        y_pred, z_pred = diffreg.predict(x_test, predict_derivs=True)

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.title(f"price training plot for caplet with expiry {start}")
        plt.plot(x_train, y_train, 'o', label='train')
        plt.plot(x_train, y_train_mdl_cpl, 'o', label='black')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.title(f"price testing plot for caplet with expiry {start}")
        plt.plot(x_test, y_pred, 'o', label='predictions')
        plt.plot(x_test, y_test_mdl_cpl, 'o', label='black')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.title(f"delta training plot for caplet with expiry {start}")
        plt.plot(x_train, z_train, 'o', label='train')
        plt.plot(x_train, z_train_mdl_cpl, 'o', label='bump and reval')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.title(f"delta testing plot for caplet with expiry {start}")
        plt.plot(x_test, z_pred, 'o', label='predictions')
        plt.plot(x_test, z_test_mdl_cpl, 'o', label='bump and reval')
        plt.legend()

        plt.tight_layout()
        plt.show()











