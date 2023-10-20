from abc import ABC, abstractmethod
import torch
from application.engine.standard_scalar import StandardScaler


def normal_equation(X: torch.Tensor, y: torch.Tensor, use_SVD: bool = True):
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    if X.dim() == 1:
        X = X.reshape(-1, 1)

    if use_SVD:
        # Use "sparse" SVD to handle matrices (X) without full rank low rank / linear in a memory efficient way
        # See https://www2.math.ethz.ch/education/bachelor/lectures/hs2014/other/linalg_INFK/svdneu.pdf
        # or LSM Reloaded page 6.
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        w = Vh.T @ torch.linalg.pinv(torch.diag(S)) @ U.T @ y
    else:
        XTX_inv = torch.linalg.pinv(X.T @ X)
        XTy = X.T @ y
        w = XTX_inv @ XTy

    return w.squeeze()


def create_polynomial_features(X: torch.tensor, deg: int = 5, bias: bool = True):
    """
    param:  X:              Matrix which size NxM. Rows are observations and columns are features
    param:  deg:            Degree of the polynomial
    param:  bias:           Add an intercept, i.e. a feature (column) where all observations (rows) are one (1)

    returns:
        A matrix with the transformed features.

    Note:
    X is assumed to be a matrix with dim (N, M),
        where each row is an observation.
    The features generated are
        [1, x[i], x[i]**2, x[i]**3, ... x[i]**deg].
    After the transformation, all the features are stacked horizontally.
    """
    if X.dim() == 1:
        X = X.reshape(-1, 1)
    X_pow = torch.hstack([torch.pow(X, i) for i in range(1, deg + 1)])
    if bias:
        one = torch.ones(size=(len(X), 1))
        phi = torch.hstack([one, X_pow])
    else:
        phi = X_pow

    return phi


class OLSRegressor(ABC):
    @property
    @abstractmethod
    def coef(self) -> torch.Tensor:
        pass

    @abstractmethod
    def set_coef(self, coef: torch.Tensor):
        pass

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor):
        pass


class PolynomialRegressor(OLSRegressor):
    def __init__(self, deg: int = 5, use_SVD: bool = True, bias: bool = True):
        """
        param:  deg:            Degree of the polynomial
        param:  standardize:    Standardize covariates (X) to have mean zero and variance one
        param:  use_SVD:        Use Singular Value Decomposition in the normal equation to solve for the coefficients
        param:  bias:           Add an intercept, i.e. a feature (column) where all observations (rows) are one (1)
        """
        self.deg = deg
        self.use_SVD = use_SVD
        self.bias = bias
        self._coef = None

    @property
    def coef(self):
        return self._coef

    def set_coef(self, coef: torch.Tensor):
        if coef.dim() == 1:
            coef = coef.reshape(-1, 1)
        self._coef = coef

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        phi = create_polynomial_features(X, deg=self.deg, bias=self.bias)
        self._coef = normal_equation(phi, y, self.use_SVD)

    def predict(self, X):
        phi = create_polynomial_features(X, deg=self.deg, bias=self.bias)
        return (phi @ self._coef).squeeze()
