from abc import ABC, abstractmethod
import torch


def normal_equation(X: torch.Tensor, y: torch.Tensor):
    # TODO consider using SVD as described on page 6 in "LSM reloaded" by Brian and Antoine
    ## SVD handles multi-coliniarity
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    if X.dim() == 1:
        X = X.reshape(-1, 1)



    XTX_inv = (X.T.mm(X)).inverse()
    XTy = X.T.mm(y)
    w = XTX_inv.mm(XTy)
    return w.squeeze()


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
    def __init__(self, deg: int = 5, standardize: bool = False):
        self.deg = deg
        self.standardize = standardize
        self._coef = None

    @property
    def coef(self):
        return self._coef

    def _create_features(self, X: torch.Tensor):
        """
        X is assumed to be a matrix with dim (N, M),
            where each row is an observation.
        The features generated are
            [1, x[i], x[i]**2, x[i]**3, ... x[i]**deg].
        After the transformation, all the features are stacked horizontally.
        """
        if X.dim() == 1:
            X = X.reshape(-1, 1)

        one = torch.ones(size=(len(X), 1))
        X_pow = torch.hstack([torch.pow(X, i) for i in range(1, self.deg+1)])
        if self.standardize:
            X_pow = StandardScaler().fit_transform(X_pow, dim=0)
        phi = torch.hstack([one, X_pow])

        return phi

    def set_coef(self, coef: torch.Tensor):
        if coef.dim() == 1:
            coef = coef.reshape(-1, 1)
        self._coef = coef

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        phi = self._create_features(X)
        self._coef = normal_equation(phi, y)

    def predict(self, X, idx: int = None):
        phi = self._create_features(X)
        return phi.mm(self._coef).squeeze()
