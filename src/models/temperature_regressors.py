from typing import Tuple

import torch
from torch import nn, Tensor


class TemperatureRegressor(nn.Module):
    def __init__(self, n_input_features, forward_size):
        super().__init__()
        self.first_linear = nn.Linear(n_input_features + 1, forward_size)
        self.first_activation = nn.ELU(alpha=1.0)
        self.second_linear = nn.Linear(forward_size, 1)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = self.first_activation(self.first_linear(torch.cat((x, z), dim=-1)))
        return self.second_linear(x)

class TemperatureRegressorV2(nn.Module):
    def __init__(self, n_wembed_features, dropout_rate):
        super().__init__()
        self.recurrent = nn.LSTM(1, hidden_size=n_wembed_features, batch_first=True)
        self.activation = nn.ELU(alpha=1.0)
        self.output_layer = nn.Linear(n_wembed_features, 1)

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor]):
        d, _ = self.recurrent(d, h0)
        return self.output_layer(self.activation(d))


class CustomTV2(nn.Module):
    def __init__(self, weather_embeddings_size: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.drnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        if hidden_size > 1:
            if weather_embeddings_size > 2:
                self.adapter = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(weather_embeddings_size, hidden_size), nn.Dropout(dropout_rate))
            else:
                self.adapter = nn.Sequential(nn.Linear(weather_embeddings_size, hidden_size), nn.Dropout(dropout_rate))
        else:
            if weather_embeddings_size > 2:
                self.adapter = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(weather_embeddings_size, hidden_size))
            else:
                self.adapter = nn.Sequential(nn.Linear(weather_embeddings_size, hidden_size))
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor]):
        d = self.drnn(d, self.adapter(h0[0]))[0]
        d = self.out(d)
        return d
