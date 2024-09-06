import torch
from torch import nn


class DummyInitializer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _):
        return None


class DummyInitializerV2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _):
        return None, (None, None)


class FullInitializer(nn.Module):
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, w: torch.Tensor):
        return torch.full((w.size(0), 1), self.fill_value, dtype=w.dtype, device=w.device)


class FullInitializerV2(nn.Module):
    def __init__(self, fill_value, hidden_size):
        super().__init__()
        self.fill_value = fill_value
        self.hidden_size = hidden_size

    def forward(self, w: torch.Tensor):
        zeros = torch.zeros((w.size(0), self.hidden_size), dtype=w.dtype, device=w.device)
        return torch.full((w.size(0), 1), self.fill_value, dtype=w.dtype, device=w.device), (zeros, zeros)


class LastInitializer(nn.Module):
    """Expects windowed data: (batch_size, window_size, n_features)"""

    def __init__(self, n_weather_features: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(n_weather_features, forward_size),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(forward_size, 1),
        )

    def forward(self, w):
        return self.regressor(w[:, -1, :])


class LastInitializerV2(nn.Module):
    """Expects windowed data: (batch_size, window_size, n_features)"""

    def __init__(self, n_weather_features: int, hidden_size: int, forward_size: int):
        super().__init__()
        self.z0_regressor = nn.Sequential(nn.Linear(n_weather_features, 1))
        self.h0_regressor = nn.Sequential(
            nn.Linear(n_weather_features, forward_size),
            nn.ReLU(),
            nn.Linear(forward_size, hidden_size),
        )
        self.c0_regressor = nn.Sequential(
            nn.Linear(n_weather_features, forward_size),
            nn.ReLU(),
            nn.Linear(forward_size, hidden_size),
        )

    def forward(self, w):
        return self.z0_regressor(w[:, -1, :]), (self.h0_regressor(w[:, -1, :]), self.c0_regressor(w[:, -1, :]))


class AvgInitializer(nn.Module):
    def __init__(self, n_weather_features: int, forward_size: int, dropout_rate: float):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(n_weather_features, forward_size),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(forward_size, 1),
        )

    def forward(self, w: torch.Tensor):
        return self.regressor(w.mean(dim=1))


class AvgInitializerV2(nn.Module):
    def __init__(self, n_weather_features: int, hidden_size: int, forward_size: int):
        super().__init__()
        self.z0_regressor = nn.Sequential(nn.Linear(n_weather_features, 1))
        self.h0_regressor = nn.Sequential(
            nn.Linear(n_weather_features, forward_size),
            nn.ReLU(),
            nn.Linear(forward_size, hidden_size),
        )
        self.c0_regressor = nn.Sequential(
            nn.Linear(n_weather_features, forward_size),
            nn.ReLU(),
            nn.Linear(forward_size, hidden_size),
        )

    def forward(self, w):
        avgs = w.mean(dim=1)
        return self.z0_regressor(avgs), (self.h0_regressor(avgs), self.c0_regressor(avgs))


class LSTMZ0Initializer(nn.Module):
    def __init__(self, n_weather_features: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.recurrent = nn.LSTM(n_weather_features, hidden_size, batch_first=True)
        self.output_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, w):
        w, _ = self.recurrent(w)
        return self.output_layer(w[:, -1, :])


class LSTMZ0InitializerV2(nn.Module):
    def __init__(self, n_weather_features: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.recurrent = nn.GRU(n_weather_features, hidden_size, batch_first=True)
        if hidden_size > 2:
            self.output_layer = nn.Sequential(nn.Dropout(dropout_rate), nn.ELU(), nn.Linear(hidden_size, 1))
        else:
            self.output_layer = nn.Sequential(nn.ELU(), nn.Linear(hidden_size, 1))

    def forward(self, w):
        w, h = self.recurrent(w)
        z0 = self.output_layer(w[:, -1, :])
        return z0, (h, h)

class LSTMNoZInitializerV2(nn.Module):
    def __init__(self, n_weather_features: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.recurrent = nn.LSTM(n_weather_features, hidden_size, batch_first=True)

    def forward(self, w):
        _, h = self.recurrent(w)
        return None, h

