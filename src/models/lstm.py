from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import mean_squared_error

from .pga import PGADensityLSTM
from .tools import physical_consistency, physical_inconsistency, reverse_tz_loss
from .simple_regressors import DummyInitialRegressor, SimpleInitialRegressor, LSTMDensityRegressor, TemperatureRegressor, FullInitialRegressor

class LitTZRegressor(L.LightningModule):
    def __init__(self,
                 initial_regressor: nn.Module,
                 density_regressor: nn.Module,
                 n_input_features: int,
                 initial_lr: float,
                 lr_decay_rate: float,
                 weight_decay: float,
                 density_lambda: float,
                 dropout_rate: float,
                 multiproc: bool,
                 **kwargs
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.initial_regressor = torch.compile(initial_regressor)
        self.density_regressor = torch.compile(density_regressor)
        self.temperature_regressor = torch.compile(TemperatureRegressor(n_input_features, dropout_rate), fullgraph=True)
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.density_lambda = density_lambda
        self.multiproc = multiproc

    def compute_losses(self, y_hat, y):
        t_loss = mse_loss(y_hat[..., 0], y[..., 0])
        z_loss = mse_loss(y_hat[..., 1], y[..., 1])
        return {"total": t_loss + self.density_lambda*z_loss, "t": t_loss, "z": z_loss}

    @staticmethod
    def compute_scores(y_hat, y):
        t_score = -mean_squared_error(y_hat[..., 0], y[..., 0], squared=False)
        z_score = -mean_squared_error(y_hat[..., 1], y[..., 1], squared=False)
        physics_score = physical_consistency(y_hat[..., 1], tol=1e-2, axis=1, agg_dims=(0,1)) # tol=1e-2 is approximately 1e-5 kg/m3, as the std is in the order of 1e-3
        return {"t": t_score, "z": z_score, "monotonicity": physics_score}

    def forward(self, x, w, **kwargs):
        x = self.dropout(x)
        w = self.dropout(w)
        z0_hat = self.initial_regressor(w)
        z_hat = self.density_regressor(x, z0_hat)
        t_hat = self.temperature_regressor(torch.cat([x, z_hat], dim=-1))
        return torch.cat([t_hat, z_hat], dim=-1)

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, w, y = batch
        y_hat = self(x, w)
        losses = self.compute_losses(y_hat, y)
        self.log_dict({f"train/loss/{key}": value for key, value in losses.items()}, on_step=False, on_epoch=True, sync_dist=self.multiproc)
        return losses["total"]

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, w, y = batch
        y_hat = self(x, w)
        losses = self.compute_losses(y_hat, y)
        scores = self.compute_scores(y_hat, y)
        self.log_dict({f"valid/loss/{key}": value for key, value in losses.items()} | {f"valid/score/{key}": value for key, value in scores.items()} | {"hp_metric": scores["t"]}, on_step=False, on_epoch=True, sync_dist=self.multiproc)
        return losses["total"]

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        return self(batch[0], batch[1])

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=100, min_lr=1e-4, threshold=5e-3, threshold_mode="abs")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss/total"}

class LitLSTM(LitTZRegressor):
    def __init__(self, n_input_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool):
        initial_regressor = DummyInitialRegressor()
        density_regressor = LSTMDensityRegressor(n_input_features, dropout_rate)
        super().__init__(initial_regressor, density_regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate, multiproc)
        self.save_hyperparameters()

class TheirLSTM(LitLSTM):
    def __init__(self, n_input_features: int, lr: float, weight_decay: float, density_lambda: float, multiproc: bool):
        super().__init__(n_input_features, lr, 0.999999999, weight_decay, density_lambda, dropout_rate=0.2, multiproc=multiproc)

class LitPGLLSTM(LitTZRegressor):
    def __init__(self, n_input_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, physics_penalty_lambda: float, density_lambda: float, dropout_rate: float, multiproc: bool):
        initial_regressor = DummyInitialRegressor()
        density_regressor = LSTMDensityRegressor(n_input_features, dropout_rate)
        super().__init__(initial_regressor, density_regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate, multiproc)
        self.physics_penalty_lambda = physics_penalty_lambda
        self.save_hyperparameters()

    def compute_losses(self, y_hat, y):
        normal_losses = super().compute_losses(y_hat, y)
        monotonicity_physics_loss = physical_inconsistency(y_hat[..., 1], tol=1e-2, axis=1, agg_dims=(0,1))
        normal_losses["monotonicity"] = monotonicity_physics_loss
        normal_losses["total"] += self.physics_penalty_lambda * normal_losses["monotonicity"]
        return normal_losses

class TheirPGLLSTM(LitPGLLSTM):
    def __init__(self, n_input_features: int, lr: float, weight_decay: float, density_lambda: float, physics_penalty_lambda: float, multiproc: bool):
        super().__init__(n_input_features, lr, 0.999999999, weight_decay, physics_penalty_lambda, density_lambda, dropout_rate=0.2, multiproc=multiproc)

class LitPGALSTM(LitTZRegressor):
    def __init__(self, initial_regressor: nn.Module, n_input_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, hidden_size: int, forward_size: int, density_lambda: float, dropout_rate: float, multiproc: bool):
        density_regressor = PGADensityLSTM(n_input_features, hidden_size, forward_size, dropout_rate)
        super().__init__(initial_regressor, density_regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate, multiproc)

class TheirPGALSTM(LitPGALSTM):
    def __init__(self, n_input_features: int, lr: float, weight_decay: float, density_lambda: float, multiproc: bool):
        initial_regressor = FullInitialRegressor(0.0)
        super().__init__(initial_regressor, n_input_features, lr, lr_decay_rate=0.999999999, weight_decay=weight_decay, hidden_size=8, forward_size=5, density_lambda=density_lambda, dropout_rate=0.2, multiproc=multiproc)

class MyPGALSTMZ0(LitPGALSTM):
    """PGA con regressor per stato iniziale
    """
    def __init__(self, n_input_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, hidden_size: int = 8, forward_size: int = 5, initial_regressor=None):
        if initial_regressor is None:
            initial_regressor = SimpleInitialRegressor(n_input_features, forward_size, dropout_rate)
        super().__init__(initial_regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
        self.save_hyperparameters(ignore=["initial_regressor"])

class MyPGALSTMLoss(LitPGALSTM):
    """PGA con loss extra
    """
    def __init__(self,
                 z_mean: Tensor, z_std: Tensor,
                 t_mean: Tensor, t_std: Tensor,
                 n_input_features: int,
                 initial_lr: float,
                 lr_decay_rate: float,
                 ttoz_penalty_lambda: float,
                 weight_decay: float,
                 density_lambda: float,
                 dropout_rate: float,
                 multiproc: bool,
                 initial_regressor: Optional[nn.Module] = None,
                 hidden_size: int=8,
                 forward_size: int=5,
        ):
        if not initial_regressor:
            initial_regressor = FullInitialRegressor(0.0)
        super().__init__(initial_regressor, n_input_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
        self.ttoz_penalty_lambda = ttoz_penalty_lambda
        self.t_mean = t_mean
        self.t_std = t_std
        self.z_mean = z_mean
        self.z_std = z_std
        self.save_hyperparameters(ignore=["initial_regressor"])

    def compute_losses(self, y_hat, y):
        normal_losses = super().compute_losses(y_hat, y)
        ttoz_physics_loss = reverse_tz_loss(y_hat[..., 0], y_hat[..., 1], self.z_mean, self.z_std, self.t_mean,
                                            self.t_std)
        normal_losses["ttoz"] = ttoz_physics_loss
        normal_losses["total"] += self.ttoz_penalty_lambda * ttoz_physics_loss
        return normal_losses

