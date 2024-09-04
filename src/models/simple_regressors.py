import torch
from torch import nn

class LSTMDensityRegressor(nn.Module):
    """Network estimating densities from features. Expects `Xd` as input. Output has size 1
    """
    def __init__(self, n_features, dropout_rate):
        super().__init__()
        self.n_features = n_features
        self.dropout_rate = dropout_rate
        self.lstm = nn.LSTM(n_features, hidden_size=8, num_layers=1, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(8, 5),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, _) -> torch.Tensor:
        x, _ = self.lstm(x)
        return self.dense_layers(x)

class TemperatureRegressor(nn.Sequential):
    def __init__(self, n_input_features, dropout_rate):
        super().__init__(
            nn.Linear(n_input_features + 1, 5),
            nn.Dropout(dropout_rate),
            nn.ELU(alpha=1.0),
            nn.Linear(5, 1)
        )

class DummyInitialRegressor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _):
        return None

class FullInitialRegressor(nn.Module):
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, w: torch.Tensor):
        return torch.full((w.size(0), 1), self.fill_value, dtype=w.dtype, device=w.device)

class SimpleInitialRegressor(nn.Module):
    """Expects windowed data: (batch_size, window_size, n_features)"""
    def __init__(self, n_input_features: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.n_input_features = n_input_features
        self.forward_size = forward_size
        self.dropout_rate = dropout_rate
        self.regressor = nn.Sequential(nn.Linear(n_input_features, forward_size), nn.Dropout(p=dropout_rate), nn.Linear(forward_size, 1))

    def forward(self, w):
        return self.regressor(w[:, -1, :])

class AvgInitialRegressor(nn.Module):
    def __init__(self, n_input_features: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.n_input_features = n_input_features
        self.forward_size = forward_size
        self.dropout_rate = dropout_rate
        # self.sum_idxs = sum_idxs
        # all_indices = torch.arange(self.n_input_features, dtype=torch.int32, device=self.sum_idxs.device)
        # self.avg_idx = torch.tensor([idx for idx in all_indices if idx not in self.sum_idxs], dtype=torch.int32, device=self.sum_idxs.device)
        self.regressor = nn.Sequential(nn.Linear(n_input_features, forward_size), nn.Dropout(p=dropout_rate), nn.Linear(forward_size, 1))

    def forward(self, w: torch.Tensor):
        return self.regressor(w.mean(dim=1))

class LSTMInitialRegressor(nn.Module):
    def __init__(self, n_input_features: int, hidden_size: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.n_input_features = n_input_features
        self.forward_size = forward_size
        self.dropout_rate = dropout_rate
        self.recurrent = nn.LSTM(n_input_features, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, w):
        x, _ = self.recurrent(w)
        x = self.dropout(x[:, -1, :])
        return self.output_layer(x)
