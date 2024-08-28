from itertools import chain
from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.nn.functional import mse_loss, dropout, relu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional import r2_score, mean_squared_error


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
            nn.ELU(alpha=1),
            nn.Linear(5, 1)
        )

class MCSampler(L.LightningModule):
    def __init__(self, model, sample_size):
        super().__init__()
        self.model = model
        self.sample_size = sample_size

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.model.training_step(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.model.validation_step(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.forward(*args, **kwargs)

    def sample(self, x, sample_size):
        self.train()
        results = [self(x) for _ in range(sample_size)]
        return torch.stack(results, dim=-1)

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        return self.sample(batch[0], self.sample_size)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

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
        z_loss = mse_loss(y_hat[..., 1], y[..., 1])
        t_loss = mse_loss(y_hat[..., 0], y[..., 0])
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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(),
            # [{"params": chain(*map(lambda l: l.parameters(), self.linear_layers + [self.output_layer])),
            #   "weight_decay": self.weight_decay},
            #  {"params": self.recurrent_layer.parameters()}],
            lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=100, min_lr=5e-6, threshold=5e-3, threshold_mode="abs")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss"}

class LitMCBaselineLSTM(L.LightningModule):
    def __init__(self, n_features, n_linear_layers, dropout_rate, mc_iterations, weight_decay, initial_lr, lr_decay_rate, n_recurrent_layers=1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_features = n_features
        self.mc_iterations = mc_iterations
        self.weight_decay = weight_decay
        self.n_recurrent_layers = n_recurrent_layers
        self.n_linear_layers = n_linear_layers
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        recurrent_dropout_rate = dropout_rate if n_recurrent_layers > 1 else 0
        self.recurrent_layer = nn.LSTM(self.n_features, num_layers=n_recurrent_layers, hidden_size=8, batch_first=True, dropout=recurrent_dropout_rate)
        linear1 = nn.Linear(8, 5)
        self.linear_layers = [linear1]
        for _ in range(self.n_linear_layers - 1):
            self.linear_layers.append(nn.Linear(5, 5))
        self.output_layer = nn.Linear(5, 1)
        self.save_hyperparameters()

    def forward(self, x, use_dropout):
        x = relu(dropout(self.recurrent_layer(x)[0], training=use_dropout))
        for layer in self.linear_layers:
            x = relu(dropout(layer(x), training=use_dropout))
        return self.output_layer(x)

    def sample(self, x, sample_size):
        results = [self(x, use_dropout=True) for _ in range(sample_size)]
        return torch.concatenate(results, dim=2)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        x, y = batch
        d_hat = relu(dropout(self.recurrent_layer(x)[0], training=True))
        x = d_hat
        for layer in self.linear_layers:
            x = relu(dropout(layer(x), training=True))
        T_hat = self.output_layer(x)
        # y_hat = self(x, use_dropout=True)
        T_loss = mse_loss(T_hat, y[..., ])
        self.log_dict({"train/loss": loss, "train/lr": self.lr_schedulers().get_last_lr()[-1]})
        return loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x, use_dropout=False)
        loss = mse_loss(y_hat, y)
        nrmse = -mean_squared_error(y_hat, y, squared=False)
        r2 = r2_score(y_hat.reshape((y.shape[0], -1)), y.reshape((y.shape[0], -1)))
        self.log_dict({"valid/loss": loss, "valid/score/nrmse": nrmse, "valid/score/r2": r2, "hp_metric": nrmse})
        return loss

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        return self.sample(batch[0], self.mc_iterations)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            # self.parameters(),
            [{"params": chain(*map(lambda l: l.parameters(), self.linear_layers + [self.output_layer])), "weight_decay": self.weight_decay},
            {"params": self.recurrent_layer.parameters()}],
            lr=self.initial_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=100, min_lr=5e-6, threshold=5e-3, threshold_mode="abs")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss"}

class LitMCPGLLSTM(L.LightningModule):
    def __init__(self, n_features, n_linear_layers, dropout_rate, mc_iterations, weight_decay, initial_lr, lr_decay_rate, n_recurrent_layers=1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_features = n_features
        self.mc_iterations = mc_iterations
        self.weight_decay = weight_decay
        self.n_recurrent_layers = n_recurrent_layers
        self.n_linear_layers = n_linear_layers
        self.initial_lr = initial_lr
        self.lr_decay_rate = lr_decay_rate
        recurrent_dropout_rate = dropout_rate if n_recurrent_layers > 1 else 0
        self.recurrent_layer = nn.LSTM(self.n_features, num_layers=n_recurrent_layers, hidden_size=8, batch_first=True, dropout=recurrent_dropout_rate)
        linear1 = nn.Linear(8, 5)
        self.linear_layers = [linear1]
        for _ in range(self.n_linear_layers - 1):
            self.linear_layers.append(nn.Linear(5, 5))
        self.output_layer = nn.Linear(5, 1)
        self.save_hyperparameters()

    def forward(self, x, use_dropout):
        x = relu(dropout(self.recurrent_layer(x)[0], training=use_dropout))
        for layer in self.linear_layers:
            x = relu(dropout(layer(x), training=use_dropout))
        return self.output_layer(x)

    def sample(self, x, sample_size):
        results = [self(x, use_dropout=True) for _ in range(sample_size)]
        return torch.concatenate(results, dim=2)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        x, y = batch
        # z = relu(dropout(self.recurrent_layer(x)[0], training=True))
        # x = z
        # for layer in self.linear_layers:
        #     x = relu(dropout(layer(x), training=True))
        # y_hat = self.output_layer(x)
        y_hat = self(x, use_dropout=True)
        loss = mse_loss(y_hat, y)
        self.log_dict({"train/loss": loss, "train/lr": self.lr_schedulers().get_last_lr()[-1]})
        return loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self(x, use_dropout=False)
        loss = mse_loss(y_hat, y)
        nrmse = -mean_squared_error(y_hat, y, squared=False)
        r2 = r2_score(y_hat.reshape((y.shape[0], -1)), y.reshape((y.shape[0], -1)))
        self.log_dict({"valid/loss": loss, "valid/score/nrmse": nrmse, "valid/score/r2": r2, "hp_metric": nrmse})
        return loss

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        return self.sample(batch[0], self.mc_iterations)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            # self.parameters(),
            [{"params": chain(*map(lambda l: l.parameters(), self.linear_layers + [self.output_layer])), "weight_decay": self.weight_decay},
            {"params": self.recurrent_layer.parameters()}],
            lr=self.initial_lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_decay_rate, patience=100, min_lr=5e-6, threshold=5e-3, threshold_mode="abs")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train/loss"}