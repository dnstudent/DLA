from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import mean_squared_error

from .pga import PGADensityLSTM
from .tools import physical_consistency, physical_inconsistency, reverse_tz_loss


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class LitTZRegressor(L.LightningModule):
    def __init__(self, regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate, **kwargs):
        super().__init__()
        self.density_regressor = torch.compile(regressor)
        self.temperature_regressor = torch.compile(TemperatureRegressor(n_input_features, dropout_rate), fullgraph=True)
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.density_lambda = density_lambda

    def compute_losses(self, y_hat, y):
        t_loss = mse_loss(y_hat[..., 0], y[..., 0])
        z_loss = mse_loss(y_hat[..., 1], y[..., 1])
        return {"total": t_loss + self.density_lambda*z_loss, "t": t_loss, "z": z_loss}

    @staticmethod
    def compute_scores(y_hat, y):
        t_score = -mean_squared_error(y_hat[..., 0], y[..., 0], squared=False)
        z_score = -mean_squared_error(y_hat[..., 1], y[..., 1], squared=False)
        physics_score = physical_consistency(y_hat[..., 1], tol=1e-4) # tol=1e-2 is approximately 1e-5 kg/m3, as the std is in the order of 1e-3
        return {"t": t_score, "z": z_score, "monotonicity": physics_score}

    def forward(self, x, *args, **kwargs):
        z_hat = self.density_regressor(x, *args, **kwargs)
        t_hat = self.temperature_regressor(torch.cat([x, z_hat], dim=-1))
        return torch.cat([t_hat, z_hat], dim=-1)

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x)
        losses = self.compute_losses(y_hat, y)
        self.log_dict({f"train/loss/{key}": value for key, value in losses.items()}, on_step=False, on_epoch=True)
        return losses["total"]

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x)
        losses = self.compute_losses(y_hat, y)
        scores = self.compute_scores(y_hat, y)
        self.log_dict({f"valid/loss/{key}": value for key, value in losses.items()} | {f"valid/score/{key}": value for key, value in scores.items()} | {"hp_metric": scores["t"]}, on_step=False, on_epoch=True)
        return losses["total"]

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        return self(batch[0])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=100, min_lr=1e-4, threshold=5e-3, threshold_mode="abs")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss/total"}

class LitLSTM(LitTZRegressor):
    def __init__(self, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda=5, dropout_rate=0.2):
        regressor = LSTMDensityRegressor(n_input_features, dropout_rate)
        super().__init__(regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate)
        self.save_hyperparameters("initial_lr", "lr_decay_rate", "weight_decay", "dropout_rate", "density_lambda")

class TheirLSTM(LitLSTM):
    def __init__(self, n_input_features, lr, weight_decay):
        super().__init__(n_input_features, lr, 0.999999999, weight_decay, 5, 0.2)

class LitPGLLSTM(LitTZRegressor):
    def __init__(self, n_input_features, initial_lr, lr_decay_rate, weight_decay, physics_penalty_lambda, density_lambda=5, dropout_rate=0.2):
        regressor = LSTMDensityRegressor(n_input_features, dropout_rate)
        super().__init__(regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate)
        self.physics_penalty_lambda = physics_penalty_lambda
        self.save_hyperparameters("initial_lr", "lr_decay_rate", "weight_decay", "dropout_rate", "density_lambda", "physics_penalty_lambda")

    def compute_losses(self, y_hat, y):
        normal_losses = super().compute_losses(y_hat, y)
        monotonicity_physics_loss = physical_inconsistency(y_hat[..., 1], tol=1e-4)
        normal_losses["monotonicity"] = monotonicity_physics_loss
        normal_losses["total"] += self.physics_penalty_lambda * normal_losses["monotonicity"]
        return normal_losses

class TheirPGLLSTM(LitPGLLSTM):
    def __init__(self, n_input_features, lr, weight_decay, physics_penalty_lambda):
        super().__init__(n_input_features, lr, 0.999999999, weight_decay, physics_penalty_lambda, 5, 0.2)

class LitPGLLSTMv2(LitTZRegressor):
    def __init__(self, z_mean, z_std, t_mean, t_std, n_input_features, initial_lr, lr_decay_rate, weight_decay, physics_penalty_lambda, ttoz_penalty_lambda, density_lambda=5, dropout_rate=0.2):
        regressor = LSTMDensityRegressor(n_input_features, dropout_rate)
        super().__init__(regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate)
        self.physics_penalty_lambda = physics_penalty_lambda
        self.ttoz_penalty_lambda = ttoz_penalty_lambda
        self.t_mean = t_mean
        self.t_std = t_std
        self.z_mean = z_mean
        self.z_std = z_std
        self.save_hyperparameters("initial_lr", "lr_decay_rate", "weight_decay", "dropout_rate", "density_lambda", "ttoz_penalty_lambda", "physics_penalty_lambda")

    def compute_losses(self, y_hat, y):
        normal_losses = super().compute_losses(y_hat, y)
        monotonicity_physics_loss = physical_inconsistency(y_hat[..., 1], tol=1e-4)
        ttoz_physics_loss = reverse_tz_loss(y_hat[..., 0], y_hat[..., 1], self.z_mean, self.z_std, self.t_mean, self.t_std)
        normal_losses["monotonicity"] = monotonicity_physics_loss
        normal_losses["ttoz"] = ttoz_physics_loss
        normal_losses["total"] += self.physics_penalty_lambda * normal_losses["monotonicity"] + self.ttoz_penalty_lambda * normal_losses["ttoz"]
        return normal_losses

class LitPGALSTM(LitTZRegressor):
    # Using the values used by the paper's authors as defaults
    def __init__(self, n_input_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate):
        density_regressor = PGADensityLSTM(n_input_features, hidden_size, forward_size, dropout_rate)
        super().__init__(density_regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate)
        self.save_hyperparameters("initial_lr", "lr_decay_rate", "weight_decay", "dropout_rate", "density_lambda")

    def forward(self, x, h0):
        return super().forward(x, h0)

class TheirPGALSTM(LitPGALSTM):
    def __init__(self, n_input_features, lr, weight_decay, density_lambda):
        super().__init__(n_input_features, lr, lr_decay_rate=0.999999999, weight_decay=weight_decay, hidden_size=8, forward_size=5, density_lambda=density_lambda, dropout_rate=0.2)

    def forward(self, x):
        zeros = torch.zeros((x.size(0), self.density_regressor.density_layer.cell.hidden_size), dtype=x.dtype,
                            device=x.device)
        h0 = (zeros.clone(), zeros.clone(), torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device))
        return super().forward(x, h0)

class MyPGALSTMZ0(LitPGALSTM):
    """PGA con regressor per stato iniziale
    """
    def __init__(self, n_input_features, initial_lr, lr_decay_rate, weight_decay=0.05, hidden_size=8, forward_size=5, density_lambda=5, dropout_rate=0.2, initial_regressor=None):
        super().__init__(n_input_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate)
        if initial_regressor is None:
            initial_regressor = nn.Linear(n_input_features, 1)
        self.initial_regressor = torch.compile(initial_regressor)
        self.save_hyperparameters("initial_lr", "lr_decay_rate", "weight_decay", "dropout_rate", "density_lambda")

    def forward(self, x):
        z0_hat = self.initial_regressor(x[:, 0, :])
        zeros = torch.zeros((x.size(0), self.density_regressor.density_layer.cell.hidden_size), dtype=x.dtype, device=x.device)
        h0 = (zeros.clone(), zeros.clone(), z0_hat)
        z_hat = self.density_regressor(x, h0)
        t_hat = self.temperature_regressor(torch.cat([x, z_hat], dim=-1))
        return torch.cat([t_hat, z_hat], dim=-1)

class MyPGALSTMLoss(LitPGALSTM):
    """PGA con loss extra
    """
    def __init__(self, z_mean, z_std, t_mean, t_std, n_input_features, initial_lr, lr_decay_rate, ttoz_penalty_lambda, weight_decay=0.05, hidden_size=8, forward_size=5, density_lambda=5, dropout_rate=0.2):
        super().__init__(n_input_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate)
        self.ttoz_penalty_lambda = ttoz_penalty_lambda
        self.t_mean = t_mean
        self.t_std = t_std
        self.z_mean = z_mean
        self.z_std = z_std
        self.save_hyperparameters("initial_lr", "lr_decay_rate", "weight_decay", "dropout_rate", "density_lambda", "ttoz_penalty_lambda")

    def compute_losses(self, y_hat, y):
        normal_losses = super().compute_losses(y_hat, y)
        ttoz_physics_loss = reverse_tz_loss(y_hat[..., 0], y_hat[..., 1], self.z_mean, self.z_std, self.t_mean,
                                            self.t_std)
        normal_losses["ttoz"] = ttoz_physics_loss
        normal_losses["total"] += self.ttoz_penalty_lambda * ttoz_physics_loss
        return normal_losses

    def forward(self, x):
        zeros = torch.zeros((x.size(0), self.density_regressor.density_layer.cell.hidden_size), dtype=x.dtype, device=x.device)
        h0 = (zeros.clone(), zeros.clone(), torch.full((x.size(0), 1), -1, dtype=x.dtype, device=x.device))
        z_hat = self.density_regressor(x, h0)
        t_hat = self.temperature_regressor(torch.cat([x, z_hat], dim=-1))
        return torch.cat([t_hat, z_hat], dim=-1)

class LitPGALSTMv2(MyPGALSTMLoss):
    def __init__(self, z_mean, z_std, t_mean, t_std, n_input_features, initial_lr, lr_decay_rate, ttoz_penalty_lambda, weight_decay=0.05, hidden_size=8, forward_size=5, density_lambda=5, dropout_rate=0.2, initial_regressor=None):
        super().__init__( z_mean, z_std, t_mean, t_std, n_input_features, initial_lr, lr_decay_rate, ttoz_penalty_lambda, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate)
        if initial_regressor is None:
            initial_regressor = nn.Linear(n_input_features, 1)
        self.initial_regressor = torch.compile(initial_regressor)

    def forward(self, x):
        z0_hat = self.initial_regressor(x[:, 0, :])
        zeros = torch.zeros((x.size(0), self.density_regressor.density_layer.cell.hidden_size), dtype=x.dtype, device=x.device)
        h0 = (zeros.clone(), zeros.clone(), z0_hat)
        z_hat = self.density_regressor(x, h0)
        t_hat = self.temperature_regressor(torch.cat([x, z_hat], dim=-1))
        return torch.cat([t_hat, z_hat], dim=-1)

# class LSTMZ0Regressor(nn.Module):
#     def __init__(self, n_input_features, hidden_size, dropout_rate):
#         super(LSTMZ0Regressor, self).__init__()
#         self.recurrent_layer = nn.LSTM(n_input_features, hidden_size=hidden_size, batch_first=True)
#         self.output_layer = nn.Linear(hidden_size, 1)
#         self.dropout = nn.Dropout(p=dropout_rate)
#
#     def forward(self, x):
#         _, (x, _) = self.recurrent_layer(x)
#         x = self.dropout(x)
#         return self.output_layer(x)

# class SuperPGALSTM(MyPGALSTMZ0):
#     def __init__(self, n_input_features, initial_lr, lr_decay_rate, z0_hidden_size, weight_decay=0.05, hidden_size=8, forward_size=5, density_lambda=5, dropout_rate=0.2):
#         z0_regressor = LSTMZ0Regressor(n_input_features, z0_hidden_size, dropout_rate)
#         super().__init__(n_input_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, initial_regressor=z0_regressor)
#         self.save_hyperparameters()

class WithExtraLoss(LitTZRegressor):
    pass