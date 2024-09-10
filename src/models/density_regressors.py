from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from click.core import batch
from torch import nn, Tensor

from .pga import MonotonicLSTM


class DensityRegressor(ABC):
    @abstractmethod
    def forward(self, x: Tensor, z0: Optional[Tensor]) -> Tensor:
        pass

class DensityRegressorV2(ABC):
    @abstractmethod
    def forward(self, x: Tensor, h0: Tuple[Tensor, Tensor], z0: Optional[Tensor]) -> Tensor:
        pass

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
        x, _ = self.lstm(x)
        return self.dense_layers(x)

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


class MonotonicDensityRegressor(DensityRegressor, nn.Module):
    def __init__(self, n_input_features: int, hidden_size: int, forward_size: int, dropout_rate: float, n_delta_layers: int):
        super().__init__()
        self.net = MonotonicLSTM(n_input_features, hidden_size, forward_size, dropout_rate, n_delta_layers)

    def forward(self, w: Tensor, z0: Tensor) -> Tensor:
        zeros = torch.zeros((1, w.size(0), self.net.monotonic_layer.cell.hidden_size), dtype=w.dtype, device=w.device)
        h0 = (zeros, zeros, z0[None, ...])
        return self.net(w, h0)[0]


class MonotonicDensityRegressorV2(DensityRegressorV2, nn.Module):
    def __init__(self, n_depth_features: int, weather_embeddings_size: int, hidden_size: int, forward_size: int, dropout_rate: float, n_delta_layers: int):
        super().__init__()
        self.net = MonotonicLSTM(n_depth_features, hidden_size, forward_size, dropout_rate, n_delta_layers)
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