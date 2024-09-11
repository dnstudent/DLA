from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn.functional import dropout, relu

from .pga import MonotonicLSTM, TheirMonotonicLSTMCell, MonotonicLSTMCell


class DensityRegressor(ABC):
    @abstractmethod
    def forward(self, x: Tensor, z0: Optional[Tensor]) -> Tensor:
        pass

class DensityRegressorV2(ABC):
    @abstractmethod
    def forward(self, x: Tensor, h0: Tuple[Tensor, Tensor], z0: Optional[Tensor]) -> Tensor:
        pass

class TheirDensityRegressor(DensityRegressor, nn.Module):
    def __init__(self, n_features: int, dropout_rate: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 8, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(8, 5),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, x: Tensor, _: Optional[Tensor]) -> Tensor:
        x = self.lstm(x)[0]
        return self.dense_layers(relu(x))

class LSTMDensityRegressor(DensityRegressor, nn.Module):
    """Network estimating densities from features. Expects `Xd` as input. Output has size 1
    """
    def __init__(self, n_features, hidden_size, forward_size, dropout_rate, batch_first):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=batch_first)
        self.dense_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, forward_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(forward_size, forward_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(forward_size, 1)
        )

    def forward(self, x: Tensor, z0: Optional[Tensor]) -> Tensor:
        x = dropout(x, p=self.dropout_rate, training=True)
        x, _ = self.lstm(x)
        x = dropout(relu(x), p=self.dropout_rate, training=True)
        x = self.dense1(x)
        x = dropout(relu(x), p=self.dropout_rate, training=True)
        x = self.dense2(x)
        x = dropout(relu(x), p=self.dropout_rate, training=True)
        return self.out(x)

class LSTMDensityRegressorV2(DensityRegressorV2, nn.Module):
    def __init__(self, n_depth_features, hidden_size, forward_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(n_depth_features, hidden_size, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, forward_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(forward_size, forward_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(forward_size, 1)
        )

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor], z0: Optional[Tensor]) -> Tensor:
        d, _ = self.lstm(d, h0)
        return self.dense_layers(d)

class TheirMonotonicRegressor(DensityRegressor, nn.Module):
    def __init__(self, n_input_features: int, dropout_rate: float):
        super().__init__()
        self.net = MonotonicLSTM(n_input_features, hidden_size=8, forward_size=5, dropout_rate=dropout_rate, cell=TheirMonotonicLSTMCell)
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor, z0: Tensor) -> Tensor:
        zeros = torch.zeros((1, x.size(0), 8), dtype=x.dtype, device=x.device)
        h0 = (zeros, zeros, z0)
        return self.net(x, h0)[0]

class MonotonicRegressor(DensityRegressor, nn.Module):
    def __init__(self, n_input_features: int, hidden_size: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.net = MonotonicLSTM(n_input_features, hidden_size, forward_size, dropout_rate, cell=MonotonicLSTMCell)
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, z0: Tensor) -> Tensor:
        zeros = torch.zeros((1, x.size(0), self.hidden_size), dtype=x.dtype, device=x.device)
        h0 = (zeros, zeros, z0)
        return self.net(x, h0)[0]


class MonotonicDensityRegressorV2(DensityRegressorV2, nn.Module):
    def __init__(self, n_depth_features: int, weather_embeddings_size: int, hidden_size: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.net = MonotonicLSTM(n_depth_features, hidden_size, forward_size, dropout_rate)
        if hidden_size > 1:
            if weather_embeddings_size > 1:
                self.hadapter = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(weather_embeddings_size, hidden_size), nn.Dropout(dropout_rate))
                self.cadapter = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(weather_embeddings_size, hidden_size), nn.Dropout(dropout_rate))
            else:
                self.hadapter = nn.Sequential(nn.Linear(weather_embeddings_size, hidden_size), nn.Dropout(dropout_rate))
                self.cadapter = nn.Sequential(nn.Linear(weather_embeddings_size, hidden_size), nn.Dropout(dropout_rate))
        else:
            if weather_embeddings_size > 1:
                self.hadapter = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(weather_embeddings_size, hidden_size))
                self.cadapter = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(weather_embeddings_size, hidden_size))
            else:
                self.hadapter = nn.Sequential(nn.Linear(weather_embeddings_size, hidden_size))
                self.cadapter = nn.Sequential(nn.Linear(weather_embeddings_size, hidden_size))
        self.zadapter = nn.Linear(1, 1)

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor], z0: Tensor) -> Tensor:
        h0, c0 = h0
        h0 = self.hadapter(h0)
        c0 = self.cadapter(c0)
        z0 = self.zadapter(z0)
        return self.net(d, (h0, c0, z0))[0]