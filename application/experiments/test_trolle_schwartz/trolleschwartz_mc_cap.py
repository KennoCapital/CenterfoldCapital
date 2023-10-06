from application.engine.mcBase import mcSim, RNG
from application.engine.products import Cap
from application.engine.trolleSchwartz import trolleSchwartz
import torch

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

seed = 1234

N = 5000

measure = 'risk_neutral'

kappa = torch.tensor(0.0553)
sigma = torch.tensor(0.3325)
alpha0 = torch.tensor(0.0045)
alpha1 = torch.tensor(0.0131)
gamma = torch.tensor(0.3341)
rho = torch.tensor(0.4615)
theta = torch.tensor(0.7542)

x0 = torch.tensor([0.0])
v0= torch.tensor([0.0])
phi1_0 = torch.tensor([0.0])
phi2_0 = torch.tensor([0.0])
phi3_0 = torch.tensor([0.0])
phi4_0 = torch.tensor([0.0])
phi5_0 = torch.tensor([0.0])
phi6_0 = torch.tensor([0.0])

firstFixingDate = torch.tensor(0.25)
lastFixingDate = torch.tensor(0.75)
delta = torch.tensor(0.25)

strike = torch.tensor(0.084)

dTL = torch.linspace(0.0, lastFixingDate + delta, int(50 * (lastFixingDate + delta) + 1))

model = trolleSchwartz(gamma, kappa, theta, rho, alpha0, alpha1, x0, v0, phi1_0, phi2_0, phi3_0, phi4_0, phi5_0, phi6_0)

rng = RNG(seed=seed, simDim=2, use_av=True)

prd = Cap(
    strike=strike,
    firstFixingDate=firstFixingDate,
    lastFixingDate=lastFixingDate,
    delta=delta
)

t_event_dates = torch.concat([prd.timeline, (lastFixingDate + delta).view(1)])

cashflows = mcSim(prd, model, rng, N, dTL, simDim=2)
print('Cashflows: \n', cashflows)

payoff = torch.sum(cashflows, dim=0)
print('Payoffs:\n', payoff)

mc_price = torch.mean(payoff)
print('MC Price =', mc_price)

#print('Model price =', model.calc_cap(r0, t_event_dates, delta, strike))
