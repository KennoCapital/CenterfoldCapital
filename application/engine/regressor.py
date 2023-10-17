from abc import ABC, abstractmethod
import torch


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self._eps = 1e-12

    def fit(self, X: torch.Tensor, dim: int = 0):
        self.mean = torch.mean(X, dim=dim)
        self.std = torch.maximum(torch.std(X, dim=dim), torch.tensor(self._eps))

    def transform(self, X: torch.Tensor):
        return (X - self.mean) / self.std

    def fit_transform(self, X: torch.Tensor, dim: int = 0):
        self.fit(X, dim)
        return self.transform(X)


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


def create_polynomial_features(X: torch.tensor, deg: int = 5, standardize: bool = True, bias: bool = True):
    """
    param:  X:              Matrix which size NxM. Rows are observations and columns are features
    param:  deg:            Degree of the polynomial
    param:  standardize:    Standardize the features to have
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
    if standardize:
        X_pow = StandardScaler().fit_transform(X_pow, dim=0)
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
    def __init__(self, deg: int = 5, standardize: bool = False, use_SVD: bool = True, bias: bool = True):
        """
        param:  deg:            Degree of the polynomial
        param:  standardize:    Standardize covariates (X) to have mean zero and variance one
        param:  use_SVD:        Use Singular Value Decomposition in the normal equation to solve for the coefficients
        param:  bias:           Add an intercept, i.e. a feature (column) where all observations (rows) are one (1)
        """
        self.deg = deg
        self.standardize = standardize
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
        phi = create_polynomial_features(X, deg=self.deg, standardize=self.standardize, bias=self.bias)
        self._coef = normal_equation(phi, y, self.use_SVD)

    def predict(self, X):
        phi = create_polynomial_features(X, deg=self.deg, standardize=self.standardize, bias=self.bias)
        return (phi @ self._coef).squeeze()


class DifferentialPolynomialRegressor(OLSRegressor):
    def __init__(self,
                 deg: int = 5,
                 alpha: float = 1.0,
                 standardize: bool = False,
                 use_SVD: bool = True,
                 bias: bool = True):
        """
        PyTorch version of Differential Polynomial Regressor inspired by
        https://github.com/differential-machine-learning/notebooks/blob/master/DifferentialRegression.ipynb

        param:  deg:            Degree of the polynomial
        param:  alpha:          Coefficient of Differential Regularization
        param:  standardize:    Standardize covariates (X) to have mean zero and variance one
        param:  use_SVD:        Use Singular Value Decomposition in the normal equation to solve for the coefficients
        param:  bias:           Add an intercept, i.e. a feature (column) where all observations (rows) are one (1)
        """
        self.deg = deg
        self.alpha = alpha
        self.standardize = standardize
        self.use_SVD = use_SVD
        self.bias = bias
        self._coef = None
        self._eps = 1e-12
        self._powers = torch.arange(start=int(not self.bias), end=self.deg + 1)

    @property
    def coef(self):
        return self._coef

    def set_coef(self, coef: torch.Tensor):
        if coef.dim() == 1:
            coef = coef.reshape(-1, 1)
        self._coef = coef

    def fit(self, X: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        phi = create_polynomial_features(X, deg=self.deg, standardize=self.standardize, bias=self.bias)
        dphi = phi[:, :, None] * self._powers[None, :, :] / (X[:, None, :] + self._eps)

        lamj = (torch.pow(y, 2.0).mean(dim=0) / torch.pow(z, 2).mean(dim=0)).reshape(1, 1, -1)
        dphiw = dphi * lamj

        phiTphi = torch.tensordot(dphiw, dphi, dims=([0, 2], [0, 2]))
        phiTz = torch.tensordot(dphiw, z, dims=([0, 2], [0, 1])).reshape(-1, 1)

        inv = torch.linalg.pinv(phi.T @ phi + self.alpha * phiTphi, hermitian=True)
        self._coef = (inv @ (phi.T @ y + self.alpha * phiTz)).reshape(-1, 1)

    def predict(self, X, predict_derivs: bool = True):
        phi = create_polynomial_features(X, deg=self.deg, standardize=self.standardize, bias=self.bias)
        y_pred = (phi @ self._coef).squeeze()

        if predict_derivs:
            dphi = phi[:, :, None] * self._powers[None, :, :] / (X[:, None, :] + self._eps)
            z_pred = torch.tensordot(dphi, self._coef, dims=(1, 0)).reshape(dphi.shape[0], -1)
            return y_pred, z_pred
        else:
            return y_pred
