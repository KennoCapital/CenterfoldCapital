import torch
import itertools
from abc import ABC, abstractmethod
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


def polynomial_powers(deg: int, n_features: int, include_interactions: bool = False):
    if include_interactions:
        comb = itertools.combinations_with_replacement
        iter = itertools.chain.from_iterable(
            comb(range(n_features), i) for i in range(1, deg + 1)
        )

        powers = torch.vstack(
            [torch.bincount(torch.tensor(c), minlength=n_features) for c in iter]
        )
    else:
        powers = torch.tensor(
            [[p for _ in range(n_features)] for p in range(1, deg + 1)]
        )

    return powers


def create_polynomial_features(
        X: torch.tensor,
        deg: int = 5,
        bias: bool = True,
        include_interactions: bool = False
):
    """
    param:  X:              Matrix which size NxM. Rows are observations and columns are features
    param:  deg:            Degree of the polynomial
    param:  bias:           Add an intercept, i.e. a feature (column) where all observations (rows) are one (1)
    param:  include_interactions:
                            Include interactions between the monomials for all features

    returns:
        A matrix with the transformed features.
        A matrix with the powers

    Note:
    X is assumed to be a matrix with dim (N, M),
        where each row is an observation.
    After the transformation, all the features are stacked horizontally.
    """
    if X.dim() == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    if n_features == 1 and include_interactions:
        n_features = deg
    powers = polynomial_powers(deg=deg, n_features=n_features, include_interactions=include_interactions)

    X_pow = torch.hstack([torch.prod(torch.pow(X, p), dim=1, keepdim=True) for p in powers])

    if bias:
        ones = torch.ones(size=(len(X), 1))
        phi = torch.hstack([ones, X_pow])

        zeros = torch.zeros(size=(1, powers.shape[1]))
        powers = torch.vstack([zeros, powers])
    else:
        phi = X_pow

    return phi, powers


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
    def __init__(self, deg: int = 5, use_SVD: bool = True, bias: bool = True, include_interactions: bool = False):
        """
        param:  deg:            Degree of the polynomial
        param:  standardize:    Standardize covariates (X) to have mean zero and variance one
        param:  use_SVD:        Use Singular Value Decomposition in the normal equation to solve for the coefficients
        param:  bias:           Add an intercept, i.e. a feature (column) where all observations (rows) are one (1)
        """
        self.deg = deg
        self.use_SVD = use_SVD
        self.bias = bias
        self.include_interactions = include_interactions
        self._coef = None
        self._powers = None

    @property
    def coef(self):
        return self._coef

    def set_coef(self, coef: torch.Tensor):
        if coef.dim() == 1:
            coef = coef.reshape(-1, 1)
        self._coef = coef

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        phi, self._powers = create_polynomial_features(X, deg=self.deg, bias=self.bias, include_interactions=self.include_interactions)
        self._coef = normal_equation(phi, y, self.use_SVD)

    def predict(self, X):
        phi, _ = create_polynomial_features(X, deg=self.deg, bias=self.bias, include_interactions=self.include_interactions)
        return (phi @ self._coef).squeeze()
