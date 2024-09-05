from typing import Any, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import mean_squared_error

from .density_regressors import LSTMDensityRegressor, MonotonicDensityRegressor
from .initializers import DummyInitializer, LastInitializer, AvgInitializer, FullInitializer, LSTMZ0Initializer
from .temperature_regressors import TemperatureRegressor
from .tools import physical_consistency, physical_inconsistency, reverse_tz_loss


class LitTZRegressor(L.LightningModule):
    def __init__(self,
                 initializer: nn.Module,
                 density_regressor: nn.Module,
                 n_input_features: int,
                 n_initial_features: Optional[int],
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
        self.initializer = torch.compile(initializer)
        self.density_regressor = torch.compile(density_regressor)
        self.temperature_regressor = torch.compile(TemperatureRegressor(n_input_features, forward_size=5, dropout_rate=dropout_rate), fullgraph=True)
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.density_lambda = density_lambda
        self.multiproc = multiproc
        self.n_input_features = n_input_features
        self.n_initial_features = n_initial_features

    def compute_losses(self, t_hat, z_hat, y):
        t_loss = mse_loss(t_hat.squeeze(-1), y[..., 0])
        z_loss = mse_loss(z_hat.squeeze(-1), y[..., 1])
        return {"total": t_loss + self.density_lambda*z_loss, "t": t_loss, "z": z_loss}

    @staticmethod
    def compute_scores(t_hat, z_hat, y):
        z_hat = z_hat.squeeze(-1)
        t_score = -mean_squared_error(t_hat.squeeze(-1), y[..., 0], squared=False)
        z_score = -mean_squared_error(z_hat, y[..., 1], squared=False)
        physics_score = physical_consistency(z_hat, tol=1e-2, axis=1, agg_dims=(0,1)) # tol=1e-2 is approximately 1e-5 kg/m3, as the std is in the order of 1e-3
        return {"t": t_score, "z": z_score, "monotonicity": physics_score, "sum": t_score + z_score}

    def forward(self, x, w, **kwargs):
        w = self.dropout(w)
        z0_hat = self.initializer(w)
        z_hat = self.density_regressor(self.dropout(x), z0_hat)
        t_hat = self.temperature_regressor(self.dropout(x), z_hat)
        return t_hat, z_hat

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, w, y = batch
        t_hat, z_hat = self(x, w)
        losses = self.compute_losses(t_hat, z_hat, y)
        self.log_dict({f"train/loss/{key}": value for key, value in losses.items()}, on_step=False, on_epoch=True, sync_dist=self.multiproc)
        return losses["total"]

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, w, y = batch
        t_hat, z_hat = self(x, w)
        losses = self.compute_losses(t_hat, z_hat, y)
        scores = self.compute_scores(t_hat, z_hat, y)
        self.log_dict({f"valid/loss/{key}": value for key, value in losses.items()} | {f"valid/score/{key}": value for key, value in scores.items()} | {"hp_metric": scores["t"]}, on_step=False, on_epoch=True, sync_dist=self.multiproc)
        return losses["total"]

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Tensor:
        return torch.cat(self(batch[0], batch[1]), dim=-1)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=20, min_lr=8e-6, threshold=-1e-3, threshold_mode="abs", cooldown=20)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss/total"}

class LitLSTM(LitTZRegressor):
    def __init__(self, n_input_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, **kwargs):
        initializer = DummyInitializer()
        density_regressor = LSTMDensityRegressor(n_input_features, hidden_size=8, forward_size=5, dropout_rate=dropout_rate)
        super().__init__(initializer, density_regressor, n_input_features, None, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate, multiproc)
        self.save_hyperparameters()

class TheirLSTM(LitLSTM):
    def __init__(self, n_input_features: int, initial_lr: float, weight_decay: float, density_lambda: float, multiproc: bool, **kwargs):
        super().__init__(n_input_features, initial_lr, 0.999999999, weight_decay, density_lambda, dropout_rate=0.2, multiproc=multiproc)

class LitPGL(LitTZRegressor):
    def __init__(self, n_input_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, physics_penalty_lambda: float, density_lambda: float, dropout_rate: float, multiproc: bool, **kwargs):
        initializer = DummyInitializer()
        density_regressor = LSTMDensityRegressor(n_input_features, hidden_size=8, forward_size=5, dropout_rate=dropout_rate)
        super().__init__(initializer, density_regressor, n_input_features, None, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate, multiproc)
        self.physics_penalty_lambda = physics_penalty_lambda
        self.save_hyperparameters()

    def compute_losses(self, t_hat, z_hat, y):
        normal_losses = super().compute_losses(t_hat, z_hat, y)
        monotonicity_physics_loss = physical_inconsistency(z_hat.squeeze(-1), tol=1e-2, axis=1, agg_dims=(0,1))
        normal_losses["monotonicity"] = monotonicity_physics_loss
        normal_losses["total"] += self.physics_penalty_lambda * normal_losses["monotonicity"]
        return normal_losses

class TheirPGL(LitPGL):
    def __init__(self, n_input_features: int, initial_lr: float, weight_decay: float, density_lambda: float, physics_penalty_lambda: float, multiproc: bool, **kwargs):
        super().__init__(n_input_features, initial_lr, 0.999999999, weight_decay, physics_penalty_lambda, density_lambda, dropout_rate=0.2, multiproc=multiproc)

class LitPGA(LitTZRegressor):
    def __init__(self, initializer: nn.Module, n_input_features: int, n_initial_features: Optional[int], initial_lr: float, lr_decay_rate: float, weight_decay: float, hidden_size: int, forward_size: int, density_lambda: float, dropout_rate: float, multiproc: bool, **kwargs):
        density_regressor = MonotonicDensityRegressor(n_input_features, hidden_size, forward_size, dropout_rate)
        super().__init__(initializer, density_regressor, n_input_features, n_initial_features, initial_lr, lr_decay_rate, weight_decay, density_lambda, dropout_rate, multiproc)

class TheirPGA(LitPGA):
    def __init__(self, n_input_features: int, initial_lr: float, weight_decay: float, density_lambda: float, multiproc: bool, **kwargs):
        initializer = FullInitializer(0.0)
        super().__init__(initializer, n_input_features, None, initial_lr, lr_decay_rate=0.999999999, weight_decay=weight_decay, hidden_size=8, forward_size=5, density_lambda=density_lambda, dropout_rate=0.2, multiproc=multiproc)

class PGAZ0Last(LitPGA):
    """PGA con regressor per stato iniziale
    """
    def __init__(self, n_input_features: int, n_initial_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, hidden_size: int = 8, forward_size: int = 5, initializer=None, **kwargs):
        if initializer is None:
            initializer = LastInitializer(n_initial_features, forward_size, dropout_rate)
        super().__init__(initializer, n_input_features, n_initial_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
        self.save_hyperparameters(ignore=["initializer"])

class PGAZ0Avg(LitPGA):
    """PGA con regressor per stato iniziale
    """
    def __init__(self, n_input_features: int, n_initial_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, hidden_size: int = 8, forward_size: int = 5, initializer=None, **kwargs):
        if initializer is None:
            initializer = AvgInitializer(n_initial_features, forward_size, dropout_rate)
        super().__init__(initializer, n_input_features, n_initial_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
        self.save_hyperparameters(ignore=["initializer"])

class PGAZ0RNN(LitPGA):
    """PGA con regressor per stato iniziale
    """
    def __init__(self, n_input_features: int, n_initial_features: int, initial_lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, hidden_size_initial: int = 5, hidden_size_density: int = 8, forward_size: int = 5, initializer=None, **kwargs):
        if initializer is None:
            initializer = LSTMZ0Initializer(n_initial_features, hidden_size_initial, dropout_rate)
        super().__init__(initializer, n_input_features, n_initial_features, initial_lr, lr_decay_rate, weight_decay, hidden_size_density, forward_size, density_lambda, dropout_rate, multiproc)
        self.save_hyperparameters(ignore=["initializer"])

class PGATtoZLoss(LitPGA):
    """PGA con loss extra
    """
    def __init__(self,
                 z_mean: Tensor, z_std: Tensor,
                 t_mean: Tensor, t_std: Tensor,
                 n_input_features: int,
                 n_initial_features: Optional[int],
                 initial_lr: float,
                 lr_decay_rate: float,
                 ttoz_penalty_lambda: float,
                 weight_decay: float,
                 density_lambda: float,
                 dropout_rate: float,
                 multiproc: bool,
                 initializer: Optional[nn.Module] = None,
                 hidden_size: int=8,
                 forward_size: int=5,
                 **kwargs
                 ):
        if not initializer:
            initializer = AvgInitializer(n_initial_features, forward_size, dropout_rate)
        super().__init__(initializer, n_input_features, n_initial_features, initial_lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
        self.ttoz_penalty_lambda = ttoz_penalty_lambda
        self.t_mean = t_mean
        self.t_std = t_std
        self.z_mean = z_mean
        self.z_std = z_std
        self.save_hyperparameters(ignore=["initializer"])

    def compute_losses(self, t_hat, z_hat, y):
        normal_losses = super().compute_losses(t_hat, z_hat, y)
        ttoz_physics_loss = reverse_tz_loss(t_hat.squeeze(-1), z_hat.squeeze(-1), self.z_mean, self.z_std, self.t_mean, self.t_std)
        normal_losses["ttoz"] = ttoz_physics_loss
        normal_losses["total"] += self.ttoz_penalty_lambda * ttoz_physics_loss
        return normal_losses