from typing import Any, Optional, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torchmetrics.functional import mean_squared_error

from .density_regressors import MonotonicRegressor, TheirDensityRegressor, TheirMonotonicRegressor
from .initializers import DummyInitializer, LastInitializer, AvgInitializer, FullInitializer, LSTMZ0Initializer
from .temperature_regressors import TheirTemperatureRegressor
from .tools import physical_consistency, physical_inconsistency, reverse_tz_loss


def configure_their_optimizer(density_regressor: nn.Module, temperature_regressor: nn.Module, lr, weight_decay):
    first_linear_kernel = [param for name, param in temperature_regressor.named_parameters() if name == "first_linear.weight"]
    other_params = [param for name, param in temperature_regressor.named_parameters() if name != "first_linear.weight"]
    return torch.optim.Adam([
            {"params": density_regressor.parameters()},
            {"params": first_linear_kernel, "weight_decay": weight_decay},
            {"params": other_params},
        ], lr=lr, eps=1e-7, weight_decay=0.0)

class TZRegressor(nn.Module):
    def __init__(self, initializer, density_regressor, temperature_regressor):
        super().__init__()
        self.initializer = initializer
        self.density_regressor = density_regressor
        self.temperature_regressor = temperature_regressor

    def forward(self, x: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        z0_hat = self.initializer(w)
        z = self.density_regressor(x, z0_hat)
        t = self.temperature_regressor(x, z)
        return t, z

class LitTZRegressor(L.LightningModule):
    def __init__(self,
                 initializer: nn.Module,
                 density_regressor: nn.Module,
                 temperature_regressor: nn.Module,
                 n_input_features: int,
                 n_initial_features: Optional[int],
                 x_padding: int,
                 lr: Optional[float],
                 weight_decay: float,
                 density_lambda: float,
                 multiproc: bool,
                 **kwargs
                 ):
        super().__init__()
        self.tz_regressor = torch.compile(TZRegressor(initializer, density_regressor, temperature_regressor))
        self.lr = lr
        self.weight_decay = weight_decay
        self.density_lambda = density_lambda
        self.multiproc = multiproc
        self.n_input_features = n_input_features
        self.n_initial_features = n_initial_features
        self.x_padding = x_padding

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
        x = nn.functional.pad(x, (self.x_padding, 0, 0, 0), mode="replicate")
        t, z = self.tz_regressor(x, w)
        return t[:, self.x_padding:].clone(), z[:, self.x_padding:].clone()

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
        first_linear_kernel = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
                               name == "first_linear.weight"]
        other_params = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
                        name != "first_linear.weight"]
        return torch.optim.Adam([
            {"params": self.tz_regressor.initializer.parameters()},
            {"params": self.tz_regressor.density_regressor.parameters()},
            {"params": first_linear_kernel, "weight_decay": self.weight_decay},
            {"params": other_params},
        ], lr=self.lr, eps=1e-7, weight_decay=0.0)

class LitLSTM(LitTZRegressor):
    def __init__(self, density_regressor: nn.Module, temperature_regressor: nn.Module, n_input_features: int, x_padding: int, lr: float, weight_decay: float, density_lambda: float, multiproc: bool):
        super().__init__(initializer=DummyInitializer(), density_regressor=density_regressor, temperature_regressor=temperature_regressor, n_input_features=n_input_features, n_initial_features=None, x_padding=x_padding, lr=lr, weight_decay=weight_decay, density_lambda=density_lambda, multiproc=multiproc)

class TheirLSTM(LitLSTM):
    def __init__(self, n_input_features: int, *, lr: float = 1e-3, weight_decay: float = 0.05, density_lambda: float = 1.0, dropout_rate: float = 0.2, multiproc: bool):
        density_regressor = TheirDensityRegressor(n_input_features, dropout_rate)
        temperature_regressor = TheirTemperatureRegressor(n_input_features)
        super().__init__(density_regressor=density_regressor, temperature_regressor=temperature_regressor, n_input_features=n_input_features, x_padding=10, lr=lr, weight_decay=weight_decay, density_lambda=density_lambda, multiproc=multiproc)
        self.save_hyperparameters()

class ProperDOutLSTM(LitLSTM):
    def __init__(self, n_input_features: int, *, lr: float, weight_decay: float,
                 density_lambda: float, dropout_rate: float, multiproc: bool):
        density_regressor = TheirDensityRegressor(n_input_features, dropout_rate)
        temperature_regressor = TheirTemperatureRegressor(n_input_features)
        super().__init__(density_regressor=density_regressor, temperature_regressor=temperature_regressor,
                         n_input_features=n_input_features, x_padding=10, lr=lr, weight_decay=weight_decay,
                         density_lambda=density_lambda, multiproc=multiproc)
        self.save_hyperparameters()

class LitPGL(LitTZRegressor):
    def __init__(self, density_regressor: nn.Module, temperature_regressor: nn.Module, n_input_features: int,
                 x_padding: int, lr: float, weight_decay: float, density_lambda: float, physics_penalty_lambda: float, multiproc: bool):
        super().__init__(initializer=DummyInitializer(), density_regressor=density_regressor,
                         temperature_regressor=temperature_regressor, n_input_features=n_input_features,
                         n_initial_features=None, x_padding=x_padding, lr=lr, weight_decay=weight_decay,
                         density_lambda=density_lambda, multiproc=multiproc)
        self.physics_penalty_labda = physics_penalty_lambda

    def compute_losses(self, t_hat, z_hat, y):
        normal_losses = super().compute_losses(t_hat, z_hat, y)
        monotonicity_physics_loss = physical_inconsistency(z_hat.squeeze(-1), tol=1e-2, axis=1, agg_dims=(0,1))
        normal_losses["monotonicity"] = monotonicity_physics_loss
        normal_losses["total"] += self.physics_penalty_lambda * normal_losses["monotonicity"]
        return normal_losses

class TheirPGL(LitPGL):
    def __init__(self, n_input_features: int, *, lr: float = 1e-3, weight_decay: float = 0.05,
                 density_lambda: float = 1.0, physics_penalty_lambda: float = 1.0, dropout_rate: float = 0.2, multiproc: bool):
        density_regressor = TheirDensityRegressor(n_input_features, dropout_rate)
        temperature_regressor = TheirTemperatureRegressor(n_input_features)
        super().__init__(density_regressor=density_regressor, temperature_regressor=temperature_regressor,
                         n_input_features=n_input_features, x_padding=10, lr=lr, weight_decay=weight_decay,
                         density_lambda=density_lambda, physics_penalty_lambda=physics_penalty_lambda, multiproc=multiproc)
        self.save_hyperparameters()

class TheirPGA(LitLSTM):
    def __init__(self, n_input_features: int, *, lr: float = 1e-3, weight_decay: float = 0.05,
                 density_lambda: float = 1.0, dropout_rate: float = 0.2, multiproc: bool):
        density_regressor = TheirMonotonicRegressor(n_input_features, dropout_rate)
        temperature_regressor = TheirTemperatureRegressor(n_input_features)
        super().__init__(density_regressor=density_regressor, temperature_regressor=temperature_regressor,
                         n_input_features=n_input_features, x_padding=10, lr=lr, weight_decay=weight_decay,
                         density_lambda=density_lambda, multiproc=multiproc)
        self.save_hyperparameters()

# class PGAZ0Last(LitPGA):
#     """PGA con regressor per stato iniziale
#     """
#     def __init__(self, n_input_features: int, n_initial_features: int, x_padding: int, lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, hidden_size: int = 8, forward_size: int = 5, initializer=None, **kwargs):
#         if initializer is None:
#             initializer = LastInitializer(n_initial_features, forward_size, dropout_rate)
#         super().__init__(initializer, n_input_features, n_initial_features, x_padding, lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
#         self.save_hyperparameters(ignore=["initializer"])
#
# class PGAZ0Avg(LitPGA):
#     """PGA con regressor per stato iniziale
#     """
#     def __init__(self, n_input_features: int, n_initial_features: int, x_padding: int, lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, hidden_size: int = 8, forward_size: int = 5, initializer=None, **kwargs):
#         if initializer is None:
#             initializer = AvgInitializer(n_initial_features, forward_size, dropout_rate)
#         super().__init__(initializer, n_input_features, n_initial_features, x_padding, lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
#         self.save_hyperparameters(ignore=["initializer"])
#
# class PGAZ0RNN(LitPGA):
#     """PGA con regressor per stato iniziale
#     """
#     def __init__(self, n_input_features: int, n_initial_features: int, x_padding: int, lr: float, lr_decay_rate: float, weight_decay: float, density_lambda: float, dropout_rate: float, multiproc: bool, hidden_size_initial: int = 5, hidden_size_density: int = 8, forward_size: int = 5, initializer=None, **kwargs):
#         if initializer is None:
#             initializer = LSTMZ0Initializer(n_initial_features, hidden_size_initial, dropout_rate)
#         super().__init__(initializer, n_input_features, n_initial_features, x_padding, lr, lr_decay_rate, weight_decay, hidden_size_density, forward_size, density_lambda, dropout_rate, multiproc)
#         self.save_hyperparameters(ignore=["initializer"])
#
# class PGATtoZLoss(LitPGA):
#     """PGA con loss extra
#     """
#     def __init__(self,
#                  z_mean: Tensor, z_std: Tensor,
#                  t_mean: Tensor, t_std: Tensor,
#                  n_input_features: int,
#                  n_initial_features: Optional[int],
#                  x_padding: int,
#                  lr: float,
#                  lr_decay_rate: float,
#                  ttoz_penalty_lambda: float,
#                  weight_decay: float,
#                  density_lambda: float,
#                  dropout_rate: float,
#                  multiproc: bool,
#                  initializer: Optional[nn.Module] = None,
#                  hidden_size: int=8,
#                  forward_size: int=5,
#                  **kwargs
#                  ):
#         if not initializer:
#             initializer = AvgInitializer(n_initial_features, forward_size, dropout_rate)
#         super().__init__(initializer, n_input_features, x_padding, n_initial_features, lr, lr_decay_rate, weight_decay, hidden_size, forward_size, density_lambda, dropout_rate, multiproc)
#         self.ttoz_penalty_lambda = ttoz_penalty_lambda
#         self.t_mean = t_mean
#         self.t_std = t_std
#         self.z_mean = z_mean
#         self.z_std = z_std
#         self.save_hyperparameters(ignore=["initializer"])
#
#     def compute_losses(self, t_hat, z_hat, y):
#         normal_losses = super().compute_losses(t_hat, z_hat, y)
#         ttoz_physics_loss = reverse_tz_loss(t_hat.squeeze(-1), z_hat.squeeze(-1), self.z_mean, self.z_std, self.t_mean, self.t_std)
#         normal_losses["ttoz"] = ttoz_physics_loss
#         normal_losses["total"] += self.ttoz_penalty_lambda * ttoz_physics_loss
#         return normal_losses