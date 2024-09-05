from typing import Tuple

from torch import nn, Tensor


class TemperatureRegressor(nn.Sequential):
    def __init__(self, n_input_features, forward_size, dropout_rate):
        super().__init__(
            nn.Linear(n_input_features + 1, forward_size),
            nn.Dropout(dropout_rate),
            nn.ELU(alpha=1.0),
            nn.Linear(forward_size, 1)
        )

class TemperatureRegressorV2(nn.Module):
    def __init__(self, n_wembed_features, dropout_rate):
        super().__init__()
        self.recurrent = nn.LSTM(1, hidden_size=n_wembed_features, batch_first=True)
        self.output_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(n_wembed_features, 1))

    def forward(self, d: Tensor, h0: Tuple[Tensor, Tensor]):
        d, _ = self.recurrent(d, h0)
        return self.output_layer(d[:, -1, :])
