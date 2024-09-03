import torch
from torch.utils.data import TensorDataset

from .windowed import WindowedDataset


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

