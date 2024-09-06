from typing import Tuple

import torch
from torch import nn, Tensor


class TemperatureRegressor(nn.Module):
    def __init__(self, n_input_features, forward_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input_features + 1, forward_size),
            nn.Dropout(dropout_rate),
            nn.ELU(alpha=1.0),
            nn.Linear(forward_size, 1)
        )

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        return self.net(torch.cat((x, z), dim=-1))

class TemperatureRegressorV2(nn.Module):
    def __init__(self, n_wembed_features, dropout_rate):
        super().__init__()
        self.recurrent = nn.LSTM(1, hidden_size=n_wembed_features, batch_first=True)
        self.output_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(n_wembed_features, 1))

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor]):
        d, _ = self.recurrent(d, h0)
        return self.output_layer(d)


class CustomTV2(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.drnn = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        if hidden_size > 2:
            self.adapter = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout_rate), nn.ReLU())
        else:
            self.adapter = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor]):
        d = self.drnn(d, self.adapter(h0[0]))[0]
        d = self.out(torch.square(d))
        return d
