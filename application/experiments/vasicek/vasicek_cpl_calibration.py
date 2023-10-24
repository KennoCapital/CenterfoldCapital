from application.engine.vasicek import Vasicek, calibrate_vasicek_cap
import torch
import scipy

torch.set_default_dtype(torch.float64)


def calibrate_vasicek_cpl(maturity, strikes, market_price,
                          a=0.86, b=0.09, sigma=0.02, r0=0.08, delta=torch.tensor(0.25)):
    def obj(x):
        x = torch.tensor(x)
        sigma = x[0]
        model = Vasicek(a, b, sigma)

        cap = model.calc_cpl(r0, maturity, delta, strikes)
        model_price = cap

        err = model_price - market_price
        mse = torch.linalg.norm(err) ** 2
        return mse

    initial_guess = scipy.optimize.minimize(
        fun=obj, x0=torch.tensor(sigma), method='Nelder-Mead', tol=1e-12).x

    return scipy.optimize.minimize(fun=obj, x0=torch.tensor(initial_guess), method='BFGS', tol=1e-12)



if __name__ == '__main__':
    maturities = torch.tensor(1.0)
    strikes = torch.empty_like(maturities, dtype=torch.float64)
    market_prices = torch.empty_like(strikes, dtype=torch.float64)
    model_prices = torch.empty_like(market_prices, dtype=torch.float64)
    delta = torch.tensor(0.25)

    # Market prices
    a_ = torch.tensor(0.86)
    b_ = torch.tensor(0.09)
    sigma_ = torch.tensor(0.0148)
    r0_ = torch.tensor(0.08)

    market_model = Vasicek(a_, b_, sigma_)
    swap_rate = r0_
    cap = market_model.calc_cpl(r0_, maturities, delta, swap_rate)
    strikes = swap_rate
    market_prices = cap

    """
    for i, T in enumerate(maturities):
        t = torch.linspace(start=float(delta), end=float(T), steps=int(T/delta))
        market_model = Vasicek(a_, b_, sigma_)
        swap_rate = market_model.calc_swap_rate(r0_, t, delta)
        cap = market_model.calc_cpl(r0_, t, delta, swap_rate)
        strikes[i] = swap_rate
        market_prices[i] = cap
    """

    # Calibration
    calib = calibrate_vasicek_cpl(maturities, strikes, market_prices, r0 = r0_)
    sigma = torch.tensor(calib.x)

    # Model prices
    vas = Vasicek(a_, b_, sigma)
    cap = vas.calc_cpl(r0_, maturities, delta, strikes)
    model_prices = cap

    # Error
    err = model_prices - market_prices
    print('Model price - Market price = {}'.format(err))
    print('MSE = {}'.format(float(torch.sum(err**2))))
    print('sigma = {}'.format(calib.x[0].round(4)))
