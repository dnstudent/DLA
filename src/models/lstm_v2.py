from typing import Any, Optional, Tuple

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import mean_squared_error
from typing_extensions import override

from .initializers import DummyInitializerV2, LastInitializerV2, AvgInitializerV2, FullInitializerV2, LSTMZ0InitializerV2
from .temperature_regressors import TemperatureRegressorV2
from .density_regressors import LSTMDensityRegressorV2, MonotonicDensityRegressorV2
from .tools import physical_consistency, physical_inconsistency, reverse_tz_loss

class LitTZRegressorV2(L.LightningModule):
    def __init__(self,
                 weather_preprocessor: nn.Module,
                 density_regressor: nn.Module,
                 temperature_regressor: nn.Module,
                 n_depth_features: int,
                 n_weather_features: Optional[int],
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
        self.weather_preprocessor = torch.compile(weather_preprocessor)
        self.density_regressor = torch.compile(density_regressor)
        self.temperature_regressor = torch.compile(temperature_regressor)
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.density_lambda = density_lambda
        self.multiproc = multiproc
        self.n_input_features = n_depth_features
        self.n_initial_features = n_weather_features

    def compute_losses(self, t_hat, z_hat, y):
        t_loss = mse_loss(t_hat, y[..., 0])
        z_loss = mse_loss(z_hat, y[..., 1])
        return {"total": t_loss + self.density_lambda*z_loss, "t": t_loss, "z": z_loss}

    @staticmethod
    def compute_scores(t_hat, z_hat, y):
        t_score = -mean_squared_error(t_hat, y[..., 0], squared=False)
        z_score = -mean_squared_error(z_hat, y[..., 1], squared=False)
        physics_score = physical_consistency(z_hat, tol=1e-2, axis=1, agg_dims=(0,1)) # tol=1e-2 is approximately 1e-5 kg/m3, as the std is in the order of 1e-3
        return {"t": t_score, "z": z_score, "monotonicity": physics_score}

    @override
    def forward(self, d, w) -> Tuple[Tensor, Tensor]:
        """
        Run the model on inputs d and w
        @param d: Spatial (depth) data: a tensor of shape (batch_size, n_depths, n_d_features)
        @param w: Temporal (weather) data: a tensor of shape (batch_size, n_timesteps, n_w_features)
        @return: A tuple with the estimates of temperature and density
        """
        w_embeds = self.weather_preprocessor(w)
        z_hat = self.density_regressor(d, w_embeds)
        t_hat = self.temperature_regressor(z_hat, w_embeds)
        return t_hat, z_hat

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=20, min_lr=8e-6,
                                         threshold=-1e-3, threshold_mode="abs", cooldown=20)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss/total"}
