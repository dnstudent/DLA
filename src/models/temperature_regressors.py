from typing import Tuple, Optional

import torch
from torch import nn, Tensor

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
    def __init__(self, n_depth_features: int, weather_embedding_size: int, hidden_size: int, forward_size: int, input_dropout: float, recurrent_dropout: float, z_dropout: float, forward_dropout: float):
        super().__init__()
        self.activation = nn.ELU(alpha=1.0)
        self.output_layer = nn.Linear(weather_embedding_size, 1)

    def forward(self, z: Tensor, x: Tensor, h0: Tensor):
        x = torch.cat((z, x), dim=-1)
        c0 = torch.zeros_like(h0)
        return self.recurrent(x, (h0, c0))

class LSTMTemperatureRegressorV2(nn.Module):
    def __init__(self, n_depth_features: int, weather_embedding_size: int, hidden_size: int, forward_size: int, input_dropout: float, recurrent_dropout: float, z_dropout: float, forward_dropout: float):
        super().__init__()
        # self.recurrent = nn.GRU(1 + n_depth_features, hidden_size=weather_embedding_size, batch_first=True)
        self.hadapter = nn.Sequential(nn.Dropout(p=forward_dropout), nn.Linear(weather_embedding_size, hidden_size))
        self.recurrent = MonotonicLSTM(1+n_depth_features, output_size=1, hidden_size=hidden_size, forward_size=forward_size, input_dropout=input_dropout, recurrent_dropout=recurrent_dropout, z_dropout=z_dropout, forward_dropout=forward_dropout, cell=MonotonicLSTMCell)
        # self.activation = nn.ELU(alpha=1.0)
        # self.output_layer = nn.Linear(weather_embedding_size, 1)
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z: Tensor, x: Tensor, h0: Tensor):
        x = torch.cat((z, x), dim=-1)
        c0 = torch.zeros_like(h0)
        return self.recurrent(x, (h0, c0))
