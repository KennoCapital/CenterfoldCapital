import torch
from torch.autograd.functional import jacobian
from application.engine.mcBase import mcSim, RNG, Model
from application.engine.products import Product


# compute differentials using jacobian
def computeJacobian_dCdr(prd : Product, model : Model, rng : RNG, N : int, r0 : torch.Tensor, dTL : torch.Tensor):
    def wrapper_dCdr(r0 : torch.Tensor):
        model.r0 = r0
        payoffs = mcSim(prd, model, rng, N, dTL)
        return payoffs

    jacobian_result = jacobian(wrapper_dCdr, r0, create_graph=False, strategy="reverse-mode")  # , price
    return jacobian_result

# TODO: If model does not have calc_fwd() member implement it
def computeJacobian_dFdr(model : Model, r0 : torch.Tensor, start : torch.Tensor, delta : torch.Tensor):
    def wrapper_dFdr(r0 : torch.Tensor):
        # model.r0 = r0
        return model.calc_fwd(r0, start, delta)

    return jacobian(wrapper_dFdr, r0, create_graph=False, strategy="reverse-mode")
