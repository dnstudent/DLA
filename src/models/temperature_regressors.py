from typing import Tuple, Optional

import torch
from torch import nn, Tensor

from .lstm import DropoutLSTM
from .tools import get_sequential_linear_weights, get_sequential_linear_biases
from ..models.pga import MonotonicLSTMCell
from ..models.lstm import MonotonicLSTM

class TheirTemperatureRegressor(nn.Module):
    def __init__(self, n_input_features: int):
        super().__init__()
        self.first_linear = nn.Linear(n_input_features + 1, 5)
        self.first_activation = nn.ELU(alpha=1.0)
        self.second_linear = nn.Linear(5, 1)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = torch.cat((x, z), dim=-1)
        x = self.first_linear(x)
        x = self.first_activation(x)
        return self.second_linear(x)

class FullDOutTemperatureRegressor(nn.Module):
    def __init__(self, n_input_features, forward_size, dropout_rate):
        super().__init__()
        self.first_linear = nn.Linear(n_input_features + 1, forward_size)
        self.first_activation = nn.ELU(alpha=1.0)
        self.second_linear = nn.Linear(forward_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = torch.cat((self.dropout(x), z), dim=-1)
        x = self.first_activation(self.first_linear(x))
        return self.second_linear(x)

class TemperatureRegressorV2(nn.Module):
    def __init__(self, n_depth_features: int, weather_embedding_size: int, forward_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_depth_features+weather_embedding_size+1, forward_size), nn.ELU(alpha=1.0), nn.Linear(forward_size, 1))

    @property
    def recursive_weights(self):
        return []

    @property
    def recursive_biases(self):
        return []

    @property
    def linear_weights(self):
        return get_sequential_linear_weights(self.net)

    @property
    def linear_biases(self):
        return get_sequential_linear_biases(self.net)

    def forward(self, x: Tensor, z: Tensor, wh0: Tensor) -> Tensor:
        x = torch.cat((x, z, wh0.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        return self.net(x)