"""
    PyTorch version of Differential Polynomial Regressor inspired by
    https://github.com/differential-machine-learning/notebooks/blob/master/DifferentialRegression.ipynb
"""

import torch
from application.engine.regressor import create_polynomial_features


class DifferentialPolynomialRegressor:
    def __init__(self,
                 deg: int = 5,
                 alpha: float = 1.0,
                 use_SVD: bool = True,
                 bias: bool = True,
                 include_interactions: bool = False):
        """
        param:  deg:            Degree of the polynomial
        param:  alpha:          Coefficient of Differential Regularization
        param:  use_SVD:        Use Singular Value Decomposition in the normal equation to solve for the coefficients
        param:  bias:           Add an intercept, i.e. a feature (column) where all observations (rows) are one (1)
        param:  include_interactions:
                                Include interactions between the monomials for all features
        """
        self.deg = deg
        self.alpha = alpha
        self.use_SVD = use_SVD
        self.bias = bias
        self.include_interactions = include_interactions
        self._coef = None
        self._powers = None
        self._eps = 1e-12


    @property
    def coef(self):
        return self._coef

    def set_coef(self, coef: torch.Tensor):
        if coef.dim() == 1:
            coef = coef.reshape(-1, 1)
        self._coef = coef

    def fit(self, X: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        phi, self._powers = create_polynomial_features(X, deg=self.deg, bias=self.bias, include_interactions=self.include_interactions)
        dphi = phi[:, :, None] * self._powers[None, :, :] / (X[:, None, :] + self._eps)

        lamj = (torch.pow(y, 2.0).mean(dim=0) / torch.pow(z, 2).mean(dim=0)).reshape(1, 1, -1)
        dphiw = dphi * lamj

        phiTphi = torch.tensordot(dphiw, dphi, dims=([0, 2], [0, 2]))
        phiTz = torch.tensordot(dphiw, z, dims=([0, 2], [0, 1])).reshape(-1, 1)

        inv = torch.linalg.pinv(phi.T @ phi + self.alpha * phiTphi, hermitian=True)
        self._coef = (inv @ (phi.T @ y + self.alpha * phiTz)).reshape(-1, 1)

    def predict(self, X, predict_derivs: bool = True):
        phi, _ = create_polynomial_features(X, deg=self.deg, bias=self.bias, include_interactions=self.include_interactions)
        y_pred = phi @ self._coef

        if predict_derivs:
            dphi = phi[:, :, None] * self._powers[None, :, :] / (X[:, None, :] + self._eps)
            z_pred = torch.tensordot(dphi, self._coef, dims=([1], [0])).reshape(dphi.shape[0], -1)
            return y_pred, z_pred
        else:
            return y_pred
