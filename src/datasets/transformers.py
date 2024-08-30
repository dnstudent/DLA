import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import preprocessing
from .tools import windowed
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

from .windowed import WindowedDataset


def split_on_time_discontinuity(X: np.array, date_column: int):
    dates: np.array = X[:, date_column]
    partition = np.ediff1d(dates, to_begin=1.0) - 1.0 > 0
    return X[~partition].reshape(-1, *X.shape[1:]), X[partition].reshape(-1, *X.shape[1:])


class WindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def transform(self, X):
        X = np.concatenate([X[:self.window_size - 1], X])
        return windowed(X, self.window_size)

    def fit(self, X, y=None):
        return self

class ContiguousTimeWindowTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        window_shape = (self.window_size, X.shape[1])
        X = np.concatenate([X[:self.window_size-1], X])
        x_left, x_right = split_on_time_discontinuity(X, 0)
        if len(x_left) > 0:
            x_left = sliding_window_view(x_left, window_shape).squeeze(axis=1)
        else:
            x_left = x_left.reshape(*((0,) + window_shape))
        if len(x_right) > 0:
            x_right = sliding_window_view(x_right, window_shape).squeeze(axis=1)
        else:
            x_right = x_right.reshape(*((0,) + window_shape))

        return np.concatenate([
            x_left, x_right
        ])

class WindowScaler(BaseEstimator, TransformerMixin):
    def __init__(self, *, with_mean=True, with_std=True):
        self.scaler = preprocessing.StandardScaler(with_mean=with_mean, with_std=with_std)
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        return self

    def transform(self, X, *args, **kwargs):
        orig_shape = X.shape
        return (
            self.scaler.transform(X.reshape(-1, X.shape[-1]), *args, **kwargs).reshape(orig_shape)
        )

    def inverse_transform(self, X, *args, **kwargs):
        orig_shape = X.shape
        return self.scaler.inverse_transform(X.reshape(-1, X.shape[-1]), *args, **kwargs).reshape(orig_shape)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)
    #     return self.transform(X, y)

class StandardScaler(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.window_size = None

    def fit(self, X):
        self.std, self.mean = torch.std_mean(X[:, :], dim=0)
        return self

    def transform(self, X):
        return (X - self.mean[None, None, :]) / self.std[None, None, :]

    def inverse_transform(self, X):
        return X*self.std[None, None, :] + self.mean[None, None, :]

def scale_wds(scaler, ds: WindowedDataset):
    return WindowedDataset(TensorDataset(scaler.transform(ds[:][0])), ds.times)

