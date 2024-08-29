from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT


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
