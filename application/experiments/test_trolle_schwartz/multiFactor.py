import numpy as np
import torch
import matplotlib.pyplot as plt

T = 1.0  # maturity
N = 252  # number of time steps
M = 1  # number of processes
dt = T / N  # time step

# params
gamma = torch.tensor(0.75)

rho = torch.tensor(0.9)
kappa = torch.tensor(1.0)
theta = torch.tensor(.04)

sigma = torch.tensor(0.3)

dWf = torch.normal(mean=0.0, std=np.sqrt(dt), size=(5, M))
dWv = rho * dWf + torch.sqrt(1 - rho ** 2) * torch.normal(mean=0.0, std=np.sqrt(dt), size=(N, M))

Wf = torch.cumsum(dWf, dim=0)
Wv = torch.cumsum(dWv, dim=0)

plt.figure()
plt.plot(Wf, color='blue')
plt.plot(Wv, color = 'red')
plt.title('Correlated Wiener processes')
plt.show()

x = torch.zeros(N, M)
x[0,:] = torch.tensor([0.01])

v = x.clone()
phi1, phi2, phi3, phi4, phi5, phi6 = x.clone(), x.clone(), x.clone(), x.clone(), x.clone(), x.clone()


def correlatedBrownians(Z, simDim, rho):
    """ Generate correlated Brownian motions """
    # Covariance matrix
    if simDim > 1:
        # we would need a cov matrix for each simulation dimension
        covMat = torch.empty((simDim, 2, 2))

        # but we only have two sources of randomness
        for i in range(len(rho)):
            covMat[i, 0, 1] = rho[i]
            covMat[i, 1, 0] = rho[i]
    else:
        covMat = torch.tensor([[1.0, rho],
                               [rho, 1.0]])

    print(covMat)
    # Cholesky decomposition: covMat = L x L^T
    L = torch.linalg.cholesky(covMat)
    print(L.t().size())
    # Correlated BMs: W = Z x L^T
    W = Z @ L.t()

    return W
rho = torch.tensor([rho])
#Z = torch.normal(mean=0, std=np.sqrt(dt),size=(2, N, M))
Z = torch.randn(size=(2, N, M))
W = correlatedBrownians(Z.reshape(-1,2), 1, rho) *np.sqrt(dt)

plt.figure()
plt.plot(W[:,0], color='blue')
plt.plot(W[:,1], color = 'red')
plt.title('Correlated BMs w Cholesky')
plt.show()

Wf2 = torch.cumsum(W[:,0], dim=0)
Wv2 = torch.cumsum(W[:,1], dim=0)
plt.figure()
plt.plot(Wf2, color='blue')
plt.plot(Wv2, color = 'red')
plt.title('Correlated BMs w Cholesky 2')
plt.show()



for  i in range(1, N):
    dv = kappa * (theta - v[i - 1,:]) * dt + sigma * torch.sqrt(v[i - 1,:]).t() * dWv[i,:]
    v[i,:] = v[i - 1,:] + dv

    dx = -gamma * x[i - 1,:] * dt + torch.sqrt(v[i - 1,:]).t() * dWf[i,:]
    x[i,:] = x[i-1,:] + dx

    dphi1 = (x[i-1,:] - gamma * phi1[i-1,:]) * dt
    phi1[i,:] = phi1[i-1,:] + dphi1

    dphi2 = (v[i-1,:] - gamma * phi2[i-1,:]) * dt
    phi2[i,:] = phi2[i-1,:] + dphi2

    dphi3 = (v[i-1,:] - 2 * gamma * phi3[i-1,:]) * dt
    phi3[i,:] = phi3[i-1,:] + dphi3

    dphi4 = (phi2[i-1,:] - gamma * phi4[i-1,:]) * dt
    phi4[i,:] = phi4[i-1,:] + dphi4

    dphi5 = (phi3[i-1,:] - 2 * gamma * phi5[i-1,:]) * dt
    phi5[i,:] = phi5[i-1,:] + dphi5

    dphi6 = (2 * phi5[i-1,:] - 2 * gamma * phi6[i-1,:]) * dt
    phi6[i,:] = phi6[i-1,:] + dphi6

plt.figure()
plt.plot(x, color='blue', label='x')
plt.plot(v, color = 'red', label='v')
plt.legend()
plt.title('state vars x & v')
plt.show()


plt.figure()
plt.plot(phi1, color = 'green', label='phi1')
plt.plot(phi2, color = 'orange', label='phi2')
plt.plot(phi3, color = 'purple', label='phi3')
plt.plot(phi4, color = 'yellow', label='phi4')
plt.plot(phi5, color = 'black', label='phi5')
plt.plot(phi6, color = 'pink', label='phi6')
plt.legend()
plt.title('state vars phis')
plt.show()


def Bx(gamma, alpha0, alpha1, t, T):
    return alpha1 / gamma * ( (1/gamma + alpha0/alpha1) * (torch.exp(-gamma*(T-t)) - 1) + (T-t) * torch.exp(-gamma*(T-t)) )

def Bphi1(gamma, alpha1, t,T):
    return alpha1 / gamma * ( torch.exp(-gamma*(T-t)) - 1 )

def Bphi2(gamma, alpha0, alpha1, t,T):
    return torch.pow(alpha1 / gamma, 2) * (1/gamma + alpha0/alpha1) * ( (1/gamma + alpha0/alpha1)* (torch.exp(-gamma*(T-t)) - 1) + (T-t) * torch.exp(-gamma*(T-t)) )
    
def Bphi3(gamma, alpha0, alpha1, t,T):
    return - alpha1 / torch.pow(gamma, 2) * ( (alpha1 / (2*torch.pow(gamma,2)) + alpha0 / gamma + torch.pow(alpha0,2) / (2*alpha1) ) * (torch.exp(-2*gamma*(T-t)) - 1) + (alpha1 / gamma + alpha0) * (T-t) * torch.exp(-2*gamma*(T-t)) + alpha1 / 2 * (T-t)**2 * torch.exp(-2*gamma*(T-t)) )

def Bphi4(gamma, alpha0, alpha1, t,T):
    return torch.pow(alpha1 / gamma,2) * (1/gamma + alpha0/alpha1) * (torch.exp(-gamma*(T-t)) - 1)
        
def Bphi5(gamma, alpha0, alpha1, t,T):
    return - alpha1 / torch.pow(gamma,2) * ( (alpha1 / gamma + alpha0) * (torch.exp(-2*gamma*(T-t)) -1) + alpha1 * (T-t) * torch.exp(-2*gamma*(T-t)) )
        
def Bphi6(gamma, alpha1, t,T):
    return - 0.5 * torch.pow(alpha1 / gamma,2) * (torch.exp(-2*gamma*(T-t)) - 1)




