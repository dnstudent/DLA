from typing import Any, Optional, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torchmetrics.functional import mean_squared_error
from typing_extensions import override

from .density_regressors import MonotonicDensityRegressorV2
from .initializers import LSTMZ0InitializerV2
from .temperature_regressors import TheirTemperatureRegressorV2
from .tools import physical_consistency


class TZRegressorV2(nn.Module):
    def __init__(self, weather_preprocessor, density_regressor, temperature_regressor):
        super().__init__()
        self.weather_preprocessor = weather_preprocessor
        self.density_regressor = density_regressor
        self.temperature_regressor = temperature_regressor

    def forward(self, x: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        z0_hat, h0 = self.weather_preprocessor(w)
        z_hat = self.density_regressor(x, h0, z0_hat)
        t_hat = self.temperature_regressor(z_hat, x, h0)
        return t_hat, z_hat

class LitTZRegressorV2(L.LightningModule):
    def __init__(self,
                 weather_preprocessor: nn.Module,
                 density_regressor: nn.Module,
                 temperature_regressor: nn.Module,
                 n_depth_features: int,
                 n_weather_features: Optional[int],
                 lr: float,
                 weight_decay: float,
                 density_lambda: float,
                 dropout_rate: float,
                 multiproc: bool,
                 **kwargs
                 ):
        super().__init__()
        self.tz_regressor = TZRegressorV2(weather_preprocessor, density_regressor, temperature_regressor)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.density_lambda = density_lambda
        self.multiproc = multiproc
        self.n_input_features = n_depth_features
        self.n_initial_features = n_weather_features

    def compute_losses(self, t_hat, z_hat, y):
        t_loss = mse_loss(t_hat.squeeze(-1), y[..., 0])
        z_loss = mse_loss(z_hat.squeeze(-1), y[..., 1])
        return {"total": t_loss + self.density_lambda*z_loss, "t": t_loss, "z": z_loss}

    @staticmethod
    def compute_scores(t_hat, z_hat, y):
        t_score = -mean_squared_error(t_hat.squeeze(-1), y[..., 0], squared=False)
        z_score = -mean_squared_error(z_hat.squeeze(-1), y[..., 1], squared=False)
        physics_score = physical_consistency(z_hat, tol=1e-2, axis=1, agg_dims=(0,1)) # tol=1e-2 is approximately 1e-5 kg/m3, as the std is in the order of 1e-3
        return {"t": t_score, "z": z_score, "monotonicity": physics_score}

    @override
    def forward(self, d: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Run the model on inputs d and w
        @param d: Spatial (depth) data: a tensor of shape (batch_size, n_depths, n_d_features)
        @param w: Temporal (weather) data: a tensor of shape (batch_size, n_timesteps, n_w_features)
        @return: A tuple with the estimates of temperature and density
        """
        return self.tz_regressor(d, w)

    @override
    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        d, w, y = batch
        t_hat, z_hat = self(d, w)
        losses = self.compute_losses(t_hat, z_hat, y)
        self.log_dict({f"train/loss/{key}": value for key, value in losses.items()}, on_step=False, on_epoch=True, sync_dist=self.multiproc)
        return losses["total"]

    @override
    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        d, w, y = batch
        t_hat, z_hat = self(d, w)
        losses = self.compute_losses(t_hat, z_hat, y)
        scores = self.compute_scores(t_hat, z_hat, y)
        self.log_dict(
            {f"valid/loss/{key}": value for key, value in losses.items()} | {f"valid/score/{key}": value for key, value
                                                                             in scores.items()} | {
                "hp_metric": scores["t"]}, on_step=False, on_epoch=True, sync_dist=self.multiproc)
        return losses["total"]

    @override
    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Tensor:
        return torch.cat(self(batch[0], batch[1]), dim=-1)

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        first_linear_kernel = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
                               name == "first_linear.weight"]
        other_params = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
                        name != "first_linear.weight"]
        return torch.optim.Adam([
            {"params": self.tz_regressor.weather_regressor.parameters()},
            {"params": self.tz_regressor.density_regressor.parameters()},
            {"params": first_linear_kernel, "weight_decay": self.weight_decay},
            {"params": other_params},
        ], lr=self.lr, eps=1e-7, weight_decay=0.0)

class SmallNet(LitTZRegressorV2):
    def __init__(self,
                 n_depth_features,
                 n_weather_features,
                 initial_lr1: float,
                 initial_lr2: float,
                 initial_lr3: float,
                 weight_decay: float,
                 density_lambda: float,
                 dropout_rate: float,
                 multiproc: bool,
                 forward_size: int,
                 weather_embedding_size: int,
                 hidden_size: int,
                 ):
        weather_preprocessor = LSTMZ0InitializerV2(n_weather_features=n_weather_features, weather_embedding_size=weather_embedding_size, dropout_rate=dropout_rate)
        density_regressor = MonotonicDensityRegressorV2(n_depth_features=n_depth_features, weather_embeddings_size=weather_embedding_size, hidden_size=hidden_size, forward_size=forward_size, dropout_rate=dropout_rate)
        temperature_regressor = TheirTemperatureRegressorV2(n_input_features=n_depth_features)
        super().__init__(weather_preprocessor=weather_preprocessor, density_regressor=density_regressor, temperature_regressor=temperature_regressor, n_depth_features=n_depth_features, n_weather_features=n_weather_features, lr=None, weight_decay=weight_decay, density_lambda=density_lambda, dropout_rate=dropout_rate, multiproc=multiproc)
        self.initial_lr1 = initial_lr1
        self.initial_lr2 = initial_lr2
        self.initial_lr3 = initial_lr3

    def configure_optimizers(self) -> OptimizerLRScheduler:
        first_linear_kernel = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
                               name == "first_linear.weight"]
        other_params = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
                        name != "first_linear.weight"]
        return torch.optim.Adam([
            {"params": self.tz_regressor.weather_regressor.parameters(), "lr": self.initial_lr1},
            {"params": self.tz_regressor.density_regressor.parameters(), "lr": self.initial_lr2},
            {"params": first_linear_kernel, "weight_decay": self.weight_decay, "lr": self.initial_lr3},
            {"params": other_params, "lr": self.initial_lr3},
        ], eps=1e-7, weight_decay=0.0)
