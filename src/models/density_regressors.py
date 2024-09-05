from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from models.pga import MonotonicLSTM


class LSTMDensityRegressor(nn.Module):
    """Network estimating densities from features. Expects `Xd` as input. Output has size 1
    """
    def __init__(self, n_features, hidden_size, forward_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
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

    def forward(self, w: Tensor, z0: Optional[Tensor]) -> Tensor:
        w, _ = self.lstm(w)
        return self.dense_layers(w)


class LSTMDensityRegressorV2(nn.Module):
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

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor], z0: Tensor) -> Tensor:
        d, _ = self.lstm(d, h0 + (z0,))
        return self.dense_layers(d)


class MonotonicDensityRegressor(nn.Module):
    def __init__(self, n_input_features, hidden_size, forward_size, dropout_rate):
        super().__init__()
        self.net = MonotonicLSTM(n_input_features, hidden_size, forward_size, dropout_rate)

    def forward(self, w: Tensor, z0: Tensor) -> Tensor:
        zeros = torch.zeros((w.size(0), self.net.monotonic_layer.cell.hidden_size), dtype=w.dtype, device=w.device)
        h0 = (zeros, zeros, z0)
        return self.net(w, h0)[0]


class MonotonicDensityRegressorV2(nn.Module):
    def __init__(self, n_depth_features, hidden_size, forward_size, dropout_rate):
        super().__init__()
        self.net = MonotonicLSTM(n_depth_features, hidden_size, forward_size, dropout_rate)

    def forward(self, d, h0, z0) -> Tensor:
        return self.net(d, h0 + (z0,))[0]