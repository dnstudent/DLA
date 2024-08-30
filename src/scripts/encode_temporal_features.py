import os
from pathlib import Path
from typing import List, Dict, Any

import lightning as L
import optuna
import polars as pl
import torch
from fontTools.misc.cython import returns
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch import optim

from src.datasets.fcrdatasets import FCRAutoencoderDataModule
from src.models.autoencoder import LitTemporalAutoencoder

if __name__ == '__main__':
    results = []
    for m in ["RNN", "LSTM"]:
        for d, r in [(1, 5), (3, 5), (3, 15), (5, 15)]:
            study = optuna.load_study(storage="postgresql://davidenicoli@localhost:5432/optuna",
                                      study_name=f"{m}_{d}d_{r}r")
            results.append(study.best_params | {"module": m, "decoder_timesteps": d, "representation_size": r,
                                                "result": study.best_value})
    best_results = pl.from_records(results).sort("decoder_timesteps", "representation_size", "result").group_by("decoder_timesteps", "representation_size", maintain_order=True).last()
    print("Best results:")
    print(best_results)
    best_results = best_results.to_dicts()

    L.seed_everything(seed=42)
    datamodule = FCRAutoencoderDataModule(os.path.join("PGA_LSTM", "Datasets", "FCR_2013_2018_Drivers.csv"), n_timesteps=7, batch_size=64, test_frac=0.05, seed=42)
    datamodule.fcr_valid = datamodule.fcr_test
    for result in best_results:
        rnn_module = getattr(nn, result["module"])
        optimizer = getattr(optim, result["optimizer_name"])
        representation_size = result["representation_size"]
        decoder_timesteps = result["decoder_timesteps"]
        lr = result["lr"]
        autoencoder = LitTemporalAutoencoder(datamodule.n_dataset_features, representation_size, 1, 7, decoder_timesteps, rnn_module, optimizer, lr)
        workdir = os.path.join("results", "fcr", "autoencoder", "best", f"{decoder_timesteps}d_{representation_size}r")
        checkpoint = ModelCheckpoint(dirpath=workdir, monitor="valid/nrmse", mode="max")
        trainer = L.Trainer(
            max_epochs=15000,
            enable_checkpointing=True,
            accelerator="cpu",
            devices=1,
            callbacks=[
                EarlyStopping(monitor="valid/nrmse", mode="max", patience=1000),
                checkpoint,
                LearningRateMonitor(logging_interval="epoch"),
            ],
            logger=TensorBoardLogger(os.path.join(workdir, "logs")),
            log_every_n_steps=10
        )
        trainer.fit(autoencoder, datamodule=datamodule)
        best_model = LitTemporalAutoencoder.load_from_checkpoint(checkpoint.best_model_path, recurrent_module=rnn_module, optimizer_class=optimizer)
        best_model.eval()
        encoded_features = (
            pl.from_numpy(
                torch.cat(trainer.predict(best_model, datamodule=datamodule), dim=0).detach().numpy(),
                schema={f"e{i}": pl.Float32 for i in range(representation_size)})
            .insert_column(0, pl.Series("time", datamodule.timesteps))
        )
        results_path = Path(os.path.join("data", "fcr", "encoded_features", f"{decoder_timesteps}d_{representation_size}r.csv"))
        results_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_features.write_csv(results_path)


