from application.engine.mcBase import mcSim, RNG
from application.engine.products import Cap
from application.engine.vasicek import Vasicek
import torch
from torch.autograd.functional import jacobian

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

seed = 1234

N = 4

a = torch.tensor(0.86)
b = torch.tensor(0.09)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08, requires_grad=True)

start = torch.tensor(0.25)
delta = torch.tensor(0.25)
expiry = torch.tensor(1.0)

eTL = torch.linspace(0.0, 1.0, 1001)    # Euler time steps

t = torch.linspace(float(delta), float(expiry), int(expiry/delta))

model = Vasicek(a, b, sigma, r0, False)
swap_rate = model.calc_swap_rate(r0, t, delta)


rng = RNG(seed=seed, use_av=True)

prd = Cap(
    strike=swap_rate,
    start=start,
    expiry=expiry,
    delta=delta
)

price, paths = mcSim(prd, model, rng, N)
print(f"Monte-Carlo cap price: {price.detach()}")
#print(model.calc_cap(r0, t, delta, swap_rate))


# compute differentials using jacobian
def computeJacobian_dCdr(prd, model, rng, N, r0):
    def wrapper_dCdr(r0):
        model.r0 = r0
        paths = mcSim(prd, model, rng, N)[1]
        return prd.payoff(paths)
    return jacobian(wrapper_dCdr, r0, create_graph=False, strategy="reverse-mode")

def computeJacobian_dFdr(model, r0):
    def wrapper_dFdr(r0):
        model.r0 = r0
        return model.calc_fwd(r0, t[1:], delta)
    return jacobian(wrapper_dFdr, r0, create_graph=False, strategy="reverse-mode")

dCdr = computeJacobian_dCdr(prd, model, rng, N, r0)
dFdr_inv = 1.0/computeJacobian_dFdr(model, r0)

# follows from chain rule
dCdF = dCdr * dFdr_inv #torch.dot(torch.pinverse(dCdr), dFdr)


"""
for i in range(N):
    print(f"pathwise differential of path {i} is: \n {J[i]}")
"""




