from application.engine.mcBase import mcSim, RNG
from application.engine.products import Cap
from application.engine.trolleSchwartz import trolleSchwartz
import torch

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

seed = 1234

N = 100

measure = 'risk_neutral'

kappa = torch.tensor(0.0553)
sigma = torch.tensor(0.3325) #0.3325 // 0.0054
alpha0 = torch.tensor(0.0045)
alpha1 = torch.tensor(0.0131)
gamma = torch.tensor(0.3341)
rho = torch.tensor(0.4615)
theta = torch.tensor(0.7542)

x0 = torch.tensor([5.0])
v0= torch.clone(x0)
phi1_0 = torch.clone(x0)
phi2_0 = torch.clone(x0)
phi3_0 = torch.clone(x0)
phi4_0 = torch.clone(x0)
phi5_0 = torch.clone(x0)
phi6_0 = torch.clone(x0)

"""
v0= torch.tensor([0.1])
phi1_0 = torch.tensor([0.1])
phi2_0 = torch.tensor([0.1])
phi3_0 = torch.tensor([0.1])
phi4_0 = torch.tensor([0.1])
phi5_0 = torch.tensor([0.1])
phi6_0 = torch.tensor([0.1])
"""

firstFixingDate = torch.tensor(0.25)
lastFixingDate = firstFixingDate #torch.tensor(0.75)
delta = torch.tensor(0.25)

strike = torch.tensor(0.084)

dTL = torch.linspace(0.0, lastFixingDate + delta, int(50 * (lastFixingDate + delta) + 1))

model = trolleSchwartz(gamma, kappa, theta, rho, sigma, alpha0, alpha1,
                       x0, v0, phi1_0, phi2_0, phi3_0, phi4_0, phi5_0, phi6_0)

rng = RNG(seed=seed, simDim=2, use_av=False)

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

mc_price = torch.nanmean(payoff)
print('MC Price =', mc_price)

#print('Model price =', model.calc_cap(r0, t_event_dates, delta, strike))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x, v, phi1, phi2, phi3, phi4, phi5, phi6 = [i for i in model.x]
    r0 = model.calc_short_rate(
        [x[:, 0, :], v[:, 0, :], phi1[:, 0, :], phi2[:, 0, :], phi3[:, 0, :], phi4[:, 0, :], phi5[:, 0, :],
         phi6[:, 0, :]], t=0.0)

    print('r0', r0.mean())

    #plot forward rates
    plt.figure()
    plt.plot(model.paths[0].fwd[0], color='blue', label='F')
    plt.plot(cashflows[0], color='red', label='cashflows')
    plt.hlines(y=strike, xmin=0, xmax=len(cashflows[0]), color='green', label='strike')
    plt.legend()
    plt.title('Forward rates')
    plt.show()

    plots = False

    if plots:
        #plot gaussians
        plt.figure()
        plt.plot( model.W[0][:,0].cumsum(dim=0)*torch.sqrt(torch.tensor(0.02)), color='blue', label='Wf')
        plt.plot( model.W[1][:,0].cumsum(dim=0)*torch.sqrt(torch.tensor(0.02)), color='red', label='Wv')
        plt.legend()
        plt.title('Correlated Wiener processes 2')
        plt.show()

        #plot stat vars
        x,v,phi1,phi2,phi3,phi4,phi5,phi6 = [i for i in model.x]

        plt.figure()
        plt.plot(x[0][:,0], color='blue', label='x')
        plt.plot(v[0][:,0], color = 'red', label='v')
        plt.legend()
        plt.title('state vars x & v')
        plt.show()

        plt.figure()
        plt.plot(phi1[0][:,0], color='blue', label='phi1')
        plt.plot(phi2[0][:,0], color = 'red', label='phi2')
        plt.plot(phi3[0][:,0], color = 'green', label='phi3')
        plt.plot(phi4[0][:,0], color = 'orange', label='phi4')
        plt.plot(phi5[0][:,0], color = 'purple', label='phi5')
        plt.plot(phi6[0][:,0], color = 'brown', label='phi6')
        plt.legend()
        plt.title('state vars phi1-6')
        plt.show()