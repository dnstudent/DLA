from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import mean_squared_error

from .tools import monotonic_loss


class DensityLSTM(nn.Module):
    """Network estimating densities from features. Expects `Xd` as input
    """
    def __init__(self, n_features, dropout_p):
        super().__init__()
        self.n_features = n_features
        self.dropout_p = dropout_p
        self.lstm = nn.LSTM(n_features, hidden_size=8, num_layers=1, batch_first=True)
        self.dense_layers = nn.Sequential(
            nn.Linear(8, 5),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self.dropout_layer = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        return self.dense_layers(x)

class TemperatureRegressor(nn.Sequential):
    def __init__(self, n_features):
        super().__init__(
            nn.Linear(n_features+1, 5),
            nn.ELU(alpha=1.0),
            nn.Linear(5, 1)
        )

class LitLSTMBaseline(L.LightningModule):
    def __init__(self, n_features, initial_lr, lr_decay_rate, weight_decay, dropout_p=0.2):
        super(LitLSTMBaseline, self).__init__()
        self.density_regressor = torch.compile(DensityLSTM(n_features, dropout_p))
        self.temperature_regressor = torch.compile(TemperatureRegressor(n_features), fullgraph=True)
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        z_hat = self.density_regressor(x)
        t_hat = self.temperature_regressor(torch.cat([x, z_hat], dim=-1))
        return torch.cat([t_hat, z_hat], dim=-1)

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x)
        t_loss = mse_loss(y_hat[..., 0], y[..., 0])
        z_loss = mse_loss(y_hat[..., 1], y[..., 1])
        loss = t_loss + z_loss
        self.log_dict({"train/loss/t": t_loss, "train/loss/z": z_loss, "train/loss": loss})
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x)
        t_loss = mse_loss(y_hat[..., 0], y[..., 0])
        t_score = -mean_squared_error(y_hat[..., 0], y[..., 0])
        z_loss = mse_loss(y_hat[..., 1], y[..., 1])
        z_score = -mean_squared_error(y_hat[..., 1], y[..., 1])
        loss = t_loss + z_loss
        self.log_dict({"valid/loss/t": t_loss, "valid/loss/z": z_loss, "valid/loss": loss, "valid/score/t": t_score, "valid/score/z": z_score, "hp_score": t_score})
        return loss

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        return self(batch[0])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            # [{"params": chain(*map(lambda l: l.parameters(), self.linear_layers + [self.output_layer])),
            #   "weight_decay": self.weight_decay},
            #  {"params": self.recurrent_layer.parameters()}],
            lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=100, min_lr=5e-6, threshold=5e-3, threshold_mode="abs")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss"}

class LitPGLLSTM(LitLSTMBaseline):
    def __init__(self, n_features, initial_lr, lr_decay_rate, weight_decay, physics_penalty_lambda, dropout_p=0.2):
        super().__init__(n_features, initial_lr, lr_decay_rate, weight_decay, dropout_p)
        self.physics_penalty_lambda = physics_penalty_lambda

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x)
        t_hat = y_hat[..., 0]
        z_hat = y_hat[..., 1]
        t_loss = mse_loss(t_hat, y[..., 0])
        z_loss = mse_loss(z_hat, y[..., 1])
        mono_loss = monotonic_loss(z_hat, ascending=True)
        loss = t_loss + z_loss + self.physics_penalty_lambda*mono_loss
        self.log_dict({
            "train/loss/t": t_loss,
            "train/loss/z": z_loss,
            "train/loss/mono": mono_loss,
            "train/loss": loss
        }, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x)
        t_hat = y_hat[..., 0]
        t = y[..., 0]
        z_hat = y_hat[..., 1]
        z = y[..., 1]
        t_loss = mse_loss(t_hat, t)
        t_score = -mean_squared_error(t_hat, t)
        z_loss = mse_loss(z_hat, z)
        z_score = -mean_squared_error(z_hat, z)
        mono_loss = monotonic_loss(z_hat, ascending=True, strict=True)
        mono_score = monotonic_loss(z_hat, ascending=False, strict=False)
        loss = t_loss + z_loss + self.physics_penalty_lambda * mono_loss
        self.log_dict({
                "valid/loss/t": t_loss,
                "valid/loss/z": z_loss,
                "valid/loss/mono": mono_loss,
                "valid/loss": loss,
                "valid/score/t": t_score,
                "valid/score/z": z_score,
                "valid/score/monotonicity": mono_score,
                "hp_metric": t_score
            })
        return loss
