from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from .pga import TheirMonotonicLSTMCell, MonotonicLSTMCell
from .lstm import MonotonicLSTM, DropoutLSTM
from .tools import get_sequential_linear_biases, get_sequential_linear_weights


class DensityRegressor(ABC):
    @abstractmethod
    def forward(self, x: Tensor, z0: Optional[Tensor]) -> Tensor:
        pass


class DensityRegressorV2(ABC):
    @abstractmethod
    def forward(
        self, x: Tensor, h0: Tuple[Tensor, Tensor], z0: Optional[Tensor]
    ) -> Tensor:
        pass


class TheirDensityRegressor(DensityRegressor, nn.Module):
    def __init__(self, n_input_features: int, dropout_rate: float):
        super().__init__()
        self.lstm = DropoutLSTM(n_input_features, hidden_size=8, dropout_rate=dropout_rate)
        # self.lstm = nn.LSTM(n_input_features, 8, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 5),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(5, 1),
        )

    def forward(self, x: Tensor, _: Optional[Tensor]) -> Tensor:
        zeros = torch.zeros((1, x.size(0), 8), dtype=x.dtype, device=x.device)
        h0 = (zeros, zeros)
        x = self.lstm(x, h0)[0]
        return self.dense_layers(x)


class TheirMonotonicRegressor(DensityRegressor, nn.Module):
    def __init__(self, n_input_features: int, dropout_rate: float):
        super().__init__()
        self.net = MonotonicLSTM(
            n_input_features, dropout=dropout_rate, cell=TheirMonotonicLSTMCell
        )
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor, z0: Tensor) -> Tensor:
        zeros = torch.zeros((1, x.size(0), 8), dtype=x.dtype, device=x.device)
        h0 = (zeros, zeros, z0)
        return self.net(x, h0)[0]


class MonotonicDensityRegressorV2(DensityRegressorV2, nn.Module):
    def __init__(
        self,
        n_depth_features: int,
        weather_embeddings_size: int,
        forward_size: int,
        hidden_size: int,
        input_dropout: float,
        recurrent_dropout: float,
        z_dropout: float,
        forward_dropout: float,
    ):
        super().__init__()
        self.net = MonotonicLSTM(
            n_input_features=n_depth_features + weather_embeddings_size,
            output_size=1,
            hidden_size=hidden_size,
            forward_size=forward_size,
            input_dropout=input_dropout,
            recurrent_dropout=recurrent_dropout,
            z_dropout=z_dropout,
            forward_dropout=forward_dropout,
            cell=MonotonicLSTMCell,
        )

    @property
    def recursive_weights(self):
        return self.net.recursive_weights

    @property
    def recursive_biases(self):
        return self.net.recursive_biases

    @property
    def linear_weights(self):
        return self.net.linear_weights

    @property
    def linear_biases(self):
        return self.net.linear_biases

    def forward(self, x: Tensor, wh0: Tensor, z0: Tensor) -> Tensor:
        # h0 = self.hadapter(h0)
        zeros = torch.zeros((1, x.size(0), self.net.cell.hidden_size), dtype=x.dtype, device=x.device)
        x = torch.cat((x, wh0.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        return self.net(x, (zeros, zeros, z0))[0]
