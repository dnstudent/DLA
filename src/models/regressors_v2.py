from itertools import chain
from typing import Any, Tuple, Optional

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor
from torch.nn.functional import mse_loss
from torchmetrics.functional import mean_squared_error
from typing_extensions import override

from .density_regressors import MonotonicDensityRegressorV2
from .initializers import LSTMZ0InitializerV2
from .temperature_regressors import TemperatureRegressorV2
from .tools import physical_consistency


class TZRegressorV2(nn.Module):
    def __init__(self, weather_preprocessor, density_regressor, temperature_regressor):
        super().__init__()
        self.weather_preprocessor = weather_preprocessor
        self.density_regressor = density_regressor
        self.temperature_regressor = temperature_regressor

    def forward(self, x: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        z0_hat, wh0 = self.weather_preprocessor(w)
        z_hat = self.density_regressor(x, wh0, z0_hat)
        t_hat = self.temperature_regressor(x, z_hat, wh0)
        return t_hat, z_hat

class LitTZRegressorV2(L.LightningModule):
    def __init__(self,
                 weather_preprocessor: nn.Module,
                 density_regressor: nn.Module,
                 temperature_regressor: nn.Module,
                 n_depth_features: int,
                 n_weather_features: int,
                 lr: Optional[float],
                 weight_decay: Optional[float],
                 density_lambda: float,
                 dropout_rate: float,
                 multiproc: bool,
                 optimizer_class,
                 ):
        super().__init__()
        self.tz_regressor = TZRegressorV2(weather_preprocessor, density_regressor, temperature_regressor)
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.density_lambda = density_lambda
        self.multiproc = multiproc
        self.n_input_features = n_depth_features
        self.n_initial_features = n_weather_features
        self.optimizer_class = optimizer_class
        self.weight_decay = weight_decay

    def compute_losses(self, t_hat, z_hat, y):
        t_loss = mse_loss(t_hat.squeeze(-1), y[..., 0])
        z_loss = mse_loss(z_hat.squeeze(-1), y[..., 1])
        return {"total": t_loss + self.density_lambda*z_loss, "t": t_loss, "z": z_loss}

    @staticmethod
    def compute_scores(t_hat, z_hat, y):
        t_score = -mean_squared_error(t_hat.squeeze(-1), y[..., 0], squared=False)
        z_score = -mean_squared_error(z_hat.squeeze(-1), y[..., 1], squared=False)
        physics_score = physical_consistency(z_hat, tol=1e-2)  # tol=1e-2 is approximately 1e-5 kg/m3, as the std is in the order of 1e-3
        return {"t": t_score, "z": z_score, "monotonicity": physics_score, "sum": t_score + z_score, }

    def forward(self, x: Tensor, w: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Run the model on inputs x and w
        @param x: Spatial (depth) data: a tensor of shape (batch_size, n_depths, n_d_features)
        @param w: Temporal (weather) data: a tensor of shape (batch_size, n_timesteps, n_w_features)
        @return: A tuple with the estimates of temperature and density
        """
        return self.tz_regressor(x, w)

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, w, y = batch
        t_hat, z_hat = self(x, w)
        losses = self.compute_losses(t_hat, z_hat, y)
        self.log_dict({f"train/loss/{key}": value for key, value in losses.items()}, on_step=False, on_epoch=True, sync_dist=self.multiproc)
        return losses["total"]

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

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Tensor:
        return torch.cat(self(batch[0], batch[1]), dim=-1)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# class SmallNet(LitTZRegressorV2):
#     def __init__(self,
#                  n_depth_features,
#                  n_weather_features,
#                  initial_lr1: float,
#                  initial_lr2: float,
#                  initial_lr3: float,
#                  recurrent_weight_decay: float,
#                  linear_weight_decay: float,
#                  density_lambda: float,
#                  hidden_dropout_rate: float,
#                  input_dropout_rate: float,
#                  multiproc: bool,
#                  forward_size: int = 5,
#                  weather_embedding_size: int = 8,
#                  hidden_size: int = 5,
#                  ):
#         weather_preprocessor = LSTMZ0InitializerV2(n_weather_features=n_weather_features, weather_embedding_size=weather_embedding_size, dropout_rate=hidden_dropout_rate)
#         density_regressor = MonotonicDensityRegressorV2(n_depth_features=n_depth_features, weather_embeddings_size=weather_embedding_size, hidden_size=hidden_size, forward_size=forward_size, input_dropout=input_dropout_rate, recurrent_dropout=hidden_dropout_rate, z_dropout=0.0, forward_dropout=hidden_dropout_rate)
#         temperature_regressor = TemperatureRegressorV2(n_depth_features=n_depth_features, weather_embedding_size=weather_embedding_size, forward_size=2, input_dropout=input_dropout_rate, recurrent_dropout=hidden_dropout_rate, z_dropout=0.0, forward_dropout=hidden_dropout_rate)
#         super().__init__(weather_preprocessor=weather_preprocessor, density_regressor=density_regressor, temperature_regressor=temperature_regressor, n_depth_features=n_depth_features, n_weather_features=n_weather_features, lr=None, weight_decay=weight_decay, density_lambda=density_lambda, dropout_rate=dropout_rate, multiproc=multiproc)
#         self.initial_lr1 = initial_lr1
#         self.initial_lr2 = initial_lr2
#         self.initial_lr3 = initial_lr3
#
#     def configure_optimizers(self) -> OptimizerLRScheduler:
#         first_linear_kernel = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
#                                name == "first_linear.weight"]
#         other_params = [param for name, param in self.tz_regressor.temperature_regressor.named_parameters() if
#                         name != "first_linear.weight"]
#         return torch.optim.Adam([
#             {"params": self.tz_regressor.weather_regressor.parameters(), "lr": self.initial_lr1},
#             {"params": self.tz_regressor.density_regressor.parameters(), "lr": self.initial_lr2},
#             {"params": first_linear_kernel, "weight_decay": self.weight_decay, "lr": self.initial_lr3},
#             {"params": other_params, "lr": self.initial_lr3},
#         ], eps=1e-7, weight_decay=0.0)

class MyNet(LitTZRegressorV2):
    def __init__(self,
                 n_depth_features,
                 n_weather_features,
                 lr1: float,
                 lr2: float,
                 lr3: float,
                 weight_decay: float,
                 density_lambda: float,
                 hidden_dropout_rate: float,
                 input_dropout_rate: float,
                 z_dropout_rate: float,
                 multiproc: bool,
                 forward_size: int,
                 hidden_size: int,
                 weather_embedding_size: int,
                 optimizer_class = torch.optim.AdamW
                 ):
        weather_preprocessor = LSTMZ0InitializerV2(n_weather_features=n_weather_features, weather_embedding_size=weather_embedding_size, dropout_rate=hidden_dropout_rate)
        density_regressor = MonotonicDensityRegressorV2(n_depth_features=n_depth_features,
                                                        weather_embeddings_size=weather_embedding_size,
                                                        hidden_size=hidden_size,
                                                        forward_size=forward_size,
                                                        input_dropout=input_dropout_rate,
                                                        recurrent_dropout=hidden_dropout_rate, z_dropout=z_dropout_rate,
                                                        forward_dropout=hidden_dropout_rate)
        temperature_regressor = TemperatureRegressorV2(n_depth_features=n_depth_features,
                                                       weather_embedding_size=weather_embedding_size,
                                                       forward_size=forward_size)
        super().__init__(weather_preprocessor=weather_preprocessor, density_regressor=density_regressor, temperature_regressor=temperature_regressor, n_depth_features=n_depth_features, n_weather_features=n_weather_features, lr=None, weight_decay=weight_decay, density_lambda=density_lambda, dropout_rate=hidden_dropout_rate, multiproc=multiproc, optimizer_class=optimizer_class)
        self.lr1 = lr1
        self.lr2 = lr2
        self.lr3 = lr3
        self.save_hyperparameters(ignore=["weather_preprocessor", "density_regressor", "temperature_regressor"])
        # self.recursive_weights = list(chain.from_iterable(map(lambda component: component.recursive_weights, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))
        # self.recursive_biases = list(chain.from_iterable(map(lambda component: component.recursive_biases, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))
        # self.linear_weights = list(chain.from_iterable(map(lambda component: component.linear_weights, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))
        # self.linear_biases = list(chain.from_iterable(map(lambda component: component.linear_biases, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))

    # @property
    # def recursive_weights(self):
    #     return list(chain.from_iterable(map(lambda component: component.recursive_weights, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))
    #
    # @property
    # def recursive_biases(self):
    #     return list(chain.from_iterable(map(lambda component: component.recursive_biases, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))
    #
    # @property
    # def linear_weights(self):
    #     return list(chain.from_iterable(map(lambda component: component.linear_weights, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))
    #
    # @property
    # def linear_biases(self):
    #     return list(chain.from_iterable(map(lambda component: component.linear_biases, [self.tz_regressor.weather_preprocessor, self.tz_regressor.density_regressor, self.tz_regressor.temperature_regressor])))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer_class(
            [
                {"params": self.tz_regressor.weather_preprocessor.parameters(), "lr": self.lr1},
                {"params": self.tz_regressor.density_regressor.parameters(), "lr": self.lr2},
                {"params": self.tz_regressor.temperature_regressor.parameters(), "lr": self.lr3},
                # {"params": self.parameters()}
             ], eps=1e-7, weight_decay=self.weight_decay,
        )