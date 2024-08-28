from typing import Any

import lightning as L
import torch
from torch import nn
from torch.nn.functional import dropout


class LitMCDropoutSequential(L.LightningModule):
    def __init__(self, layers, p, mc_iteration):
        super().__init__()
        self.layers = layers
        self.p = p
        self.mc_iteration = mc_iteration

    def forward(self, x, training):
        for layer in self.layers:
            x = layer(x)
            x = dropout(x, p=self.drouput_rate, training=training, inplace=False)
        return self.output_layer(x)

    def predict_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> Any:
        self.dropout.train()
        x, _ = batch
        predictions = [self.dropout(self.model(x)) for _ in range(self.mc_iteration)]
        return torch.concatenate(predictions, dim=-1)
