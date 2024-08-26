from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torch.nn.functional import mse_loss
from torchmetrics.functional import r2_score, mean_squared_error

class BaselineLSTM(nn.Module):
    def __init__(self, n_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_input_features = n_features
        self.recurrent = nn.LSTM(self.n_input_features, 8, batch_first=True)
        self.linear_layers = nn.Sequential(*([nn.Linear(8, 5), nn.ReLU()] + [nn.Linear(5, 5), nn.ReLU()] * 3 + [nn.Linear(5, 1)]))

    def forward(self, x, *args, **kwargs):
        x, _ = self.recurrent(x)
        return self.linear_layers(x)

class LitBaselineLSTM(L.LightningModule):
    def __init__(self, n_features):
        super().__init__()
        self.model = BaselineLSTM(n_features)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        loss = mse_loss(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        loss = mse_loss(y_hat, y)
        nrmse = -mean_squared_error(y_hat, y, squared=False)
        r2 = r2_score(y_hat.reshape((y.shape[0], -1)), y.reshape((y.shape[0], -1)))
        self.log_dict({"valid/loss": loss, "valid/score/nrmse": nrmse, "valid/score/r2": r2})
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam([{"params": self.model.linear_layers.parameters(), "weight_decay": 0.05}, {"params": self.model.recurrent.parameters()}], lr=1e-4)