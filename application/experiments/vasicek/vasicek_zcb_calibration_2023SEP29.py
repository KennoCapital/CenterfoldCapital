from application.utils.path_config import get_data_path
from application.engine.vasicek import Vasicek, calibrate_vasicek_zcb_price
from application.engine.linearProducts import zcb_yield_to_price, zcb_price_to_yield
import torch
import pandas as pd
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


if __name__ == '__main__':

    file_path_name = get_data_path('fred_zcb_yield_2023SEP29.csv')
    df = pd.read_csv(file_path_name, skiprows=1)

    t = torch.tensor(df['TERM'], dtype=torch.float64)
    y_market = torch.tensor(df['YIELD_PCT'] / 100, dtype=torch.float64)
    zcb_market = zcb_yield_to_price(t, y_market)

    calib = calibrate_vasicek_zcb_price(maturities=t, market_prices=zcb_market)
    a, b, sigma, r0 = torch.tensor(calib.x)

    vas = Vasicek(a=a, b=b, sigma=sigma, r0=r0)
    zcb_mdl = vas.calc_zcb(r0, t).flatten()
    y_mdl = zcb_price_to_yield(t, zcb_mdl)

    plt.figure()
    plt.plot(t, zcb_market, color='blue', label='Market')
    plt.scatter(t, zcb_mdl,  color='black', label='Vasicek')
    plt.legend()
    plt.title('ZCB prices')
    plt.show()

    plt.figure()
    plt.plot(t, y_market, color='blue', label='Market')
    plt.scatter(t, y_mdl, color='black', label='Vasicek')
    plt.legend()
    plt.title('ZCB yields')
    plt.show()

    print('a = {},\nb = {},\nsigma = {},\nr0 = {}'.format(*[param for param in calib.x]))

