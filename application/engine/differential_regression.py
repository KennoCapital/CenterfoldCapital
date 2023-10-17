from application.engine.AAD import computeJacobian_dCdr, computeJacobian_dFdr
from application.engine.mcBase import RNG, mcSim
from application.engine.products import Caplet
from application.engine.vasicek import Vasicek
from sklearn.preprocessing import PolynomialFeatures
import torch
import numpy as np


class DifferentialRegression: # from Savine and Huge Differential ML
    def __init__(self, degree=5, alpha=0.5):
        self.degree = degree
        self.polynomial_features = PolynomialFeatures(degree=degree, include_bias=True)
        self.alpha = alpha
        self.epsilon = 1.0e-08

    def fit(self, x, y, z):
        self.phi_ = self.polynomial_features.fit_transform(x)
        self.powers_ = self.polynomial_features.powers_
        self.dphi_ = self.phi_[:, :, np.newaxis] * self.powers_[np.newaxis, :, :] / (x[:, np.newaxis, :] + self.epsilon)
        self.lamj_ = ((y ** 2).mean(axis=0) / (z ** 2).mean(axis=0)).reshape(1, 1, -1)
        self.dphiw_ = self.dphi_ * self.lamj_
        phiTphi = np.tensordot(self.dphiw_, self.dphi_, axes=([0, 2], [0, 2]))
        phiTz = np.tensordot(self.dphiw_, z, axes=([0, 2], [0, 1])).reshape(-1, 1)
        inv = np.linalg.pinv(self.phi_.T @ self.phi_ + self.alpha * phiTphi)
        self.beta_ = (inv @ (self.phi_.T @ y + self.alpha * phiTz)).reshape(-1, 1)

    def predict(self, x, predict_derivs=False):
        phi = self.polynomial_features.transform(x)
        y_pred = phi @ self.beta_


        if predict_derivs:
            dphi = phi[:, :, np.newaxis] * self.powers_[np.newaxis, :, :] / (x[:, np.newaxis, :] + self.epsilon)
            z_pred = np.tensordot(dphi, self.beta_, (1, 0)).reshape(dphi.shape[0], -1)
            return y_pred, z_pred
        else:
            return y_pred


def diffreg_fit(prd: Caplet,
              mdl: Vasicek,
              rng:RNG,
              s: torch.Tensor,
              rs : torch.Tensor,
              measure : str = 'terminal',
              dtl : torch.Tensor = torch.tensor([])
              ):

    cprd = Caplet(
        start=prd.start - s,
        delta=prd.delta,
        strike=prd.strike
    )

    cmdl = Vasicek(mdl.a, mdl.b, mdl.sigma, rs, mdl.use_ATS, mdl.use_euler, measure)

    x_train = cmdl.calc_fwd(rs, cprd.start, cprd.delta)
    y_train = mcSim(cprd, cmdl, rng, rng.N, dtl)

    rs.requires_grad_()
    dCdr = torch.sum(computeJacobian_dCdr(cprd, cmdl, rng, rng.N, rs, dtl), dim=1)
    dFdr = torch.sum(computeJacobian_dFdr(cmdl, rs, cprd.start, cprd.delta), dim=1)

    # follows from chain rule
    z_train = dCdr * 1 / dFdr

    x_train = x_train.detach().numpy().reshape(-1, 1)
    y_train = y_train.detach().numpy().reshape(-1, 1)
    z_train = z_train.detach().numpy().reshape(-1, 1)

    diffreg = DifferentialRegression(degree=5, alpha=1.0)
    diffreg.fit(x_train, y_train, z_train)

    return diffreg, x_train, y_train, z_train
