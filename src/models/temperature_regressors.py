from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from torch.nn.functional import dropout

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
        return self.second_linear(self.dropout(x))

class TemperatureRegressorV2(nn.Module):
    def __init__(self, n_depth_features: int, weather_embedding_size: int, dropout_rate: float):
        super().__init__()
        self.recurrent = nn.GRU(1 + n_depth_features, hidden_size=weather_embedding_size, batch_first=True)
        self.activation = nn.ELU(alpha=1.0)
        self.output_layer = nn.Linear(weather_embedding_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z: Tensor, x: Tensor, h0: Tuple[Tensor, Tensor]):
        t, _ = self.recurrent(torch.cat((z, self.dropout(x)), dim=-1), h0[0])
        return self.output_layer(self.activation(t))

class TheirTemperatureRegressorV2(nn.Module):
    def __init__(self, n_input_features: int):
        super().__init__()
        self.first_linear = nn.Linear(n_input_features + 1, 5)
        self.first_activation = nn.ELU(alpha=1.0)
        self.second_linear = nn.Linear(5, 1)

    def forward(self, z:Tensor, x: Tensor, _: Optional[Tensor] = None) -> Tensor:
        x = torch.cat((x, z), dim=-1)
        x = self.first_linear(x)
        x = self.first_activation(x)
        return self.second_linear(x)


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
