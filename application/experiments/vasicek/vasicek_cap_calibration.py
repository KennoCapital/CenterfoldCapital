from application.engine.vasicek import Vasicek, calibrate_vasicek_cap
import torch

torch.set_default_dtype(torch.float64)


if __name__ == '__main__':
    maturities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]
    strikes = torch.empty_like(torch.tensor(maturities), dtype=torch.float64)
    market_prices = torch.empty_like(strikes, dtype=torch.float64)
    model_prices = torch.empty_like(market_prices, dtype=torch.float64)
    delta = 0.25

    # Market prices
    a_ = torch.tensor(0.86)
    b_ = torch.tensor(0.09)
    sigma_ = torch.tensor(0.0148)
    r0_ = torch.tensor(0.08)

    for i, T in enumerate(maturities):
        t = torch.linspace(start=delta, end=T, steps=int(T/delta))
        market_model = Vasicek(a_, b_, sigma_)
        swap_rate = market_model.calc_swap_rate(r0_, t, delta)
        cap = market_model.calc_cap(r0_, t, delta, swap_rate)
        strikes[i] = swap_rate
        market_prices[i] = cap

    # Calibration
    calib = calibrate_vasicek_cap(maturities, strikes, market_prices)
    a, b, sigma, r0 = torch.tensor(calib.x)

    # Model prices
    vas = Vasicek(a, b, sigma)
    for i, T in enumerate(maturities):
        t = torch.linspace(start=delta, end=T, steps=int(T/delta))
        cap = vas.calc_cap(r0, t, delta, strikes[i])
        model_prices[i] = cap

    # Error
    err = model_prices - market_prices
    print('Model price - Market price = {}'.format(err))
    print('MSE = {}'.format(float(torch.sum(err**2))))
    print('a = {}, b = {}, sigma = {}, r0 = {}'.format(
        calib.x[0].round(4), calib.x[1].round(4), calib.x[2].round(4), calib.x[3].round(4)
    ))
