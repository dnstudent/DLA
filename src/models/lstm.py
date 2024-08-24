import lightning as L
import torch
from torch import nn

class BaseLSTM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.LSTM())
        pass