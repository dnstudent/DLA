import torch
import lightning as L
from typing_extensions import override


class BestScore(L.Callback):
    mode_dict = {"min": torch.lt, "max": torch.gt}
    def __init__(self, monitor, mode):
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_epoch = None
        self.monitor_op = self.mode_dict[mode]

    @override
    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        logs = trainer.callback_metrics
        monitored = logs.get(self.monitor).squeeze()
        if self.best_score is None or self.monitor_op(monitored, self.best_score):
            self.best_score = monitored
            self.best_epoch = trainer.current_epoch