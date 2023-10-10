from application.engine.mcBase import lsmcDefaultSim, LSMC, RNG
from application.engine.products import BermudanPayerSwaption
from application.engine.vasicek import Vasicek
from application.engine.regressor import PolynomialRegressor
import torch

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

seed = 1234

deg = 5
n = 5000
N = 50000
notional = torch.tensor(1e6)

measure = 'risk_neutral'

a = torch.tensor(0.86)
b = torch.tensor(0.09)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)

exerciseDates = torch.tensor([5.0, 10.0, 15.0])
delta = torch.tensor(0.25)
swapFirstFixingDate = torch.tensor(0.25)
swapLastFixingDate = torch.tensor(30.0)
strike = torch.tensor(0.084)

dTL = torch.linspace(0.0, swapLastFixingDate + delta, int(50 * (swapLastFixingDate + delta) + 1))

model = Vasicek(a, b, sigma, r0, True, False, measure)

rng = RNG(seed=seed, use_av=True)

prd = BermudanPayerSwaption(
    strike=strike,
    exerciseDates=exerciseDates,
    delta=delta,
    swapFirstFixingDate=swapFirstFixingDate,
    swapLastFixingDate=swapLastFixingDate,
    notional=notional
)

poly_reg = PolynomialRegressor(deg=deg, standardize=True)
lsmc = LSMC(reg=poly_reg)

payoff = lsmcDefaultSim(
    prd=prd, mdl=model, rng=rng, N=N, n=n, lsmc=lsmc, reg=poly_reg, dTL=dTL
)

price = torch.mean(torch.sum(payoff, dim=0))

print(f'Price = {price}')
