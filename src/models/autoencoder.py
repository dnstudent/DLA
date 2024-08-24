from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchmetrics.functional import mean_squared_error, r2_score

class Encoder(nn.Module):
    def __init__(self, n_features, latent_size, num_layers, module, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.n_features = n_features
        self.latent_size = latent_size
        self.rnn = module(n_features, latent_size, num_layers, batch_first=True)

    def forward(self, x: torch.tensor, *args, **kwargs):
        x, _ = self.rnn(x)
        return x[:, -1, :] # La RNN di per sé restituisce in out gli out dei layer per ogni step temporale

class Decoder(nn.Module):
    def __init__(self, latent_size, n_features, num_layers, in_seq_length, out_seq_length, module,  *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length
        self.n_features = n_features
        self.rnn = module(latent_size, n_features, num_layers, batch_first=True)
        self.output = nn.Linear(n_features, n_features)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = x.unsqueeze(1).expand(-1, self.in_seq_length, -1)
        x, _ = self.rnn(x)
        x = self.output(x[:, -self.out_seq_length:, :])
        return x

# class Autoencoder(nn.Module):
#     def __init__(self, n_features, latent_dim, num_layers, in_seq_length, out_seq_length, recurrent_module=nn.RNN, *args, **kwargs):
#         super(Autoencoder, self).__init__(*args, **kwargs)
#         self.latent_dim = latent_dim
#         self.in_seq_len = in_seq_length
#         self.encoder = Encoder(n_features+2, latent_dim, num_layers, recurrent_module) #nn.Sequential(module(n_features, latent_dim, batch_first=True))
#         self.decoder = Decoder(latent_dim, n_features, num_layers, in_seq_length, out_seq_length, recurrent_module) #module(latent_dim, n_features, batch_first=True)
#
#     def forward(self, x, *args, **kwargs):
#         encoded = self.encoder(x, *args, **kwargs)
#         decoded = self.decoder(encoded, *args, **kwargs)
#         return decoded

class LitTemporalAutoencoder(L.LightningModule):
    def __init__(self, n_features, latent_dim, num_layers, in_seq_length, out_seq_length, recurrent_module, optimizer_class, lr, *args, **kwargs):
        super(LitTemporalAutoencoder, self).__init__(*args, **kwargs)
        # self.save_hyperparameters(ignore=["recurrent_module", "optimizer_class"])
        # Two features are the doy spec
        encoder = Encoder(n_features+2, latent_dim, num_layers, recurrent_module)
        decoder = Decoder(latent_dim, n_features, num_layers, in_seq_length, out_seq_length, recurrent_module)
        self.autoencoder = torch.compile(nn.Sequential(encoder, decoder))
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.n_features = n_features
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length

    def _common_step(self, batch, batch_idx, *args: Any, **kwargs: Any):
        x = batch[0]
        # x = x.view(x.size(0), self.in_seq_length, self.n_features)
        # z = self.encoder(x)
        # y_hat = self.decoder(z)
        y_hat = self.autoencoder(x)
        # The last two features are the doy
        y_true = x[:, -self.out_seq_length:, :-2].clone()
        return y_hat, y_true

    def training_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        y_hat, y_true = self._common_step(batch, batch_idx, *args, **kwargs)
        loss = nn.functional.mse_loss(y_hat, y_true)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        y_hat, y_true = self._common_step(batch, batch_idx, *args, **kwargs)
        loss = nn.functional.mse_loss(y_hat, y_true)
        score1 = -mean_squared_error(y_hat, y_true, squared=False)
        score2 = r2_score(y_hat.reshape((batch[0].size(0), -1)), y_true.reshape((batch[0].size(0), -1)))
        # score = r2_score(y_hat.reshape((batch[0].size(0), -1)), y_true.reshape((batch[0].size(0), -1)))
        self.log_dict({"valid/loss": loss, "valid/nrmse": score1, "valid/r2": score2, "hp_metric": score1}, on_step=False, on_epoch=True)
        # self.log("valid/score", score, on_step=False, on_epoch=True)
        return loss #{"valid/loss": loss, "valid/score": score}

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer