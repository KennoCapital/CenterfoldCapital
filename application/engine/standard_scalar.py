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


class DifferentialStandardScaler:
    def __init__(self):
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self._eps = 1e-12

    def fit(self, X: torch.Tensor, y: torch.Tensor, dim: int or tuple[int] = 0):
        if isinstance(dim, int):
            dim = (dim, dim)
        if len(dim) != 2:
            raise ValueError('Lenght of `dim` must be 2')

        self.x_mean = torch.mean(X, dim=dim[0])
        self.x_std = torch.maximum(torch.std(X, dim=dim[0]), torch.tensor(self._eps))
        self.y_mean = torch.mean(y, dim=dim[1])
        self.y_std = torch.maximum(torch.std(y, dim=dim[1]), torch.tensor(self._eps))

    def transform(self, X: torch.Tensor, y: torch.Tensor, z: torch.Tensor or None = None):
        x_pred = (X - self.x_mean) / self.x_mean
        y_pred = (y - self.y_mean) / self.y_mean

        if z is None:
            return x_pred, y_pred

        z_pred = self.x_std / self.y_std * z
        return x_pred, y_pred, z_pred

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor, z: torch.Tensor or None = None, dim: int or tuple[int] = 0):
        self.fit(X, y, dim)
        return self.transform(X, y, z)
