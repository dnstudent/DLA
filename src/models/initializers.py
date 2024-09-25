from typing import Tuple

import torch
from torch import nn, Tensor
from .lstm import DropoutLSTM, SingleDropoutLSTM
from .tools import get_sequential_linear_weights, get_sequential_linear_biases


class DummyInitializer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _):
        return None

class FullInitializer(nn.Module):
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, w: torch.Tensor):
        return torch.full((1, w.size(0), 1), self.fill_value, dtype=w.dtype, device=w.device)

# class AvgInitializerV2(nn.Module):
#     def __init__(self, n_weather_features: int, hidden_size: int, forward_size: int):
#         super().__init__()
#         self.z0_regressor = nn.Sequential(nn.Linear(n_weather_features, 1))
#         self.h0_regressor = nn.Sequential(
#             nn.Linear(n_weather_features, forward_size),
#             nn.ReLU(),
#             nn.Linear(forward_size, hidden_size),
#         )
#         self.c0_regressor = nn.Sequential(
#             nn.Linear(n_weather_features, forward_size),
#             nn.ReLU(),
#             nn.Linear(forward_size, hidden_size),
#         )
#
#     def forward(self, w):
#         avgs = w.mean(dim=1)
#         return self.z0_regressor(avgs), (self.h0_regressor(avgs), self.c0_regressor(avgs))

class LSTMZ0Initializer(nn.Module):
    def __init__(self, n_weather_features: int, weather_embedding_size: int, dropout_rate: float):
        super().__init__()
        self.recurrent = SingleDropoutLSTM(n_weather_features, weather_embedding_size, dropout_rate)
        if weather_embedding_size > 2:
            self.output_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(weather_embedding_size, 1))
        else:
            self.output_layer = nn.Sequential(nn.ReLU(), nn.Linear(weather_embedding_size, 1))

    @property
    def recursive_weights(self):
        return self.recurrent.recursive_weights

    @property
    def recursive_biases(self):
        return self.recurrent.recursive_biases

    @property
    def linear_weights(self):
        return get_sequential_linear_weights(self.output_layer)

    @property
    def linear_biases(self):
        return get_sequential_linear_biases(self.output_layer)


    def forward(self, w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        zeros = torch.zeros((w.size(0), self.recurrent.cell.hidden_size), dtype=w.dtype, device=w.device)
        h = self.recurrent(w, (zeros, zeros))
        return self.output_layer(h).unsqueeze(0)

class LSTMZ0InitializerV2(nn.Module):
    def __init__(self, n_weather_features: int, weather_embedding_size: int, dropout_rate: float):
        super().__init__()
        self.recurrent = SingleDropoutLSTM(n_weather_features, 1+weather_embedding_size, dropout_rate)
        # self.recurrent = nn.GRU(n_weather_features, weather_embedding_size, batch_first=True)
        if weather_embedding_size > 2:
            self.z_out = nn.Sequential(nn.Dropout(dropout_rate), nn.ELU(alpha=1.0), nn.Linear(1+weather_embedding_size,1))
            self.wh_out = nn.Sequential(nn.Dropout(dropout_rate), nn.ELU(alpha=1.0), nn.Linear(1+weather_embedding_size,weather_embedding_size))
        else:
            self.z_out = nn.Sequential(nn.ELU(alpha=1.0), nn.Linear(1+weather_embedding_size,1))
            self.wh_out = nn.Sequential(nn.ELU(alpha=1.0), nn.Linear(1+weather_embedding_size,weather_embedding_size))

    @property
    def recursive_weights(self):
        return self.recurrent.recursive_weights

    @property
    def recursive_biases(self):
        return self.recurrent.recursive_biases

    @property
    def linear_weights(self):
        return get_sequential_linear_weights(self.z_out) + get_sequential_linear_weights(self.wh_out)

    @property
    def linear_biases(self):
        return get_sequential_linear_biases(self.z_out) + get_sequential_linear_biases(self.wh_out)


    def forward(self, w: Tensor) -> Tuple[Tensor, Tensor]:
        zeros = torch.zeros((w.size(0), self.recurrent.cell.hidden_size), dtype=w.dtype, device=w.device)
        whh0 = self.recurrent(w, (zeros.clone(), zeros.clone()))
        return self.z_out(whh0), self.wh_out(whh0)

# class LSTMTZ0InitializerV2(nn.Module):
#     def __init__(self, n_weather_features: int, weather_embedding_size: int, dropout_rate: float):
#         super().__init__()
#         self.recurrent = DropoutLSTM(n_weather_features, weather_embedding_size, dropout_rate)
#         # self.recurrent = nn.GRU(n_weather_features, weather_embedding_size, batch_first=True)
#         if weather_embedding_size > 2:
#             self.output_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.ELU(alpha=1.0), nn.Linear(weather_embedding_size, 2))
#         else:
#             self.output_layer = nn.Sequential(nn.ELU(alpha=1.0), nn.Linear(weather_embedding_size, 2))
#
#     @property
#     def recursive_weights(self):
#         return self.recurrent.recursive_weights
#
#     @property
#     def recursive_biases(self):
#         return self.recurrent.recursive_biases
#
#     @property
#     def linear_weights(self):
#         return get_sequential_linear_weights(self.output_layer)
#
#     @property
#     def linear_biases(self):
#         return get_sequential_linear_biases(self.output_layer)
#
#
#     def forward(self, w: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
#         zeros = torch.zeros((w.size(0), self.recurrent.cell.hidden_size), dtype=w.dtype, device=w.device)
#         _, (h, _) = self.recurrent(w, (zeros.clone(), zeros.clone()))
#         tz0 = self.output_layer(h)
#         return tz0[..., 0].unsqueeze(-1), tz0[..., 1].unsqueeze(-1), h
#
# class LSTMNoZInitializerV2(nn.Module):
#     def __init__(self, n_weather_features: int, hidden_size: int, dropout_rate: float):
#         super().__init__()
#         self.recurrent = nn.LSTM(n_weather_features, hidden_size, batch_first=True)
#
#     def forward(self, w):
#         _, h = self.recurrent(w)
#         return None, h
#
