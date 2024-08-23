import argparse
import logging
import os

import lightning as L
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from sklearn.model_selection import KFold
from torch import nn
from torch.utils import data as tdata
from rich.progress import track

from src.datasets.fcrdataset import fcr_dataset
from src.datasets.transformers import StandardScaler, scale_wds
from src.datasets.windowed import WindowedDataset
from src.models.autoencoder import LitTemporalAutoencoder

WINDOW_SIZE = 7
LATENT_DIM = 5
DECODER_TIMESTEPS = 1
w = StandardScaler()

def autoencoder(rnn_module_name, optimizer_name, lr):
    rnn_module = getattr(nn, rnn_module_name)
    optimizer_class = getattr(torch.optim, optimizer_name)
    return LitTemporalAutoencoder(n_features=10,
                                  latent_dim=LATENT_DIM,
                                  num_layers=1,
                                  in_seq_length=WINDOW_SIZE,
                                  out_seq_length=DECODER_TIMESTEPS,
                                  recurrent_module=rnn_module,
                                  optimizer_class=optimizer_class,
                                  lr=lr)

def define_hparams(trial: optuna.Trial):
    rnn_module_name = trial.suggest_categorical("recurrent_layer", ["RNN", "LSTM", "GRU"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adadelta", "RMSprop", "Adam"])
    return {"rnn_module": rnn_module_name, "optimizer": optimizer_name, "lr": lr}

def objective(accelerator, max_epochs, patience, csv_path, work_dir, n_devices, profile):
    X, y, t = fcr_dataset(csv_path, WINDOW_SIZE)
    tads = WindowedDataset(tdata.TensorDataset(torch.from_numpy(X)), times=t)
    def _fn(trial: optuna.Trial):
        inner_cv = KFold(n_splits=5, shuffle=False)
        scores = []
        dsizes = []
        hparams = define_hparams(trial)
        for i, (_, val_idxs) in track(enumerate(inner_cv.split(tads)), total=inner_cv.n_splits, auto_refresh=False, transient=True):
            train_ds, val_ds = tads.train_test_split(val_idxs.copy())
            w.fit(train_ds.unique_entries(0))
            train_ds = scale_wds(w, train_ds)
            val_ds = scale_wds(w, val_ds)

            if profile:
                profiler = SimpleProfiler(dirpath=work_dir, filename="perf_logs")
            else:
                profiler = None
            trainer = L.Trainer(
                max_epochs=max_epochs,
                enable_checkpointing=False,
                accelerator=accelerator,
                devices=n_devices,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="valid/score", patience=patience, mode="max")
                ],
                logger=TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "lightning_logs"), version=f"fold_{i+1}"),
                log_every_n_steps=25,
                profiler=profiler
            )
            model = autoencoder(hparams["rnn_module"], hparams["optimizer"], hparams["lr"])

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=64, shuffle=True),
                val_dataloaders=tdata.DataLoader(val_ds, batch_size=128, shuffle=False),

            )
            trainer.logger.log_hyperparams(params=hparams, metrics=trainer.callback_metrics)
            dsizes.append(len(train_ds))
            scores.append(trainer.callback_metrics["valid/score"].item())
            trial.report(trainer.callback_metrics["valid/score"].item(), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        score = np.average(scores, weights=dsizes)
        return score
    return _fn


if __name__ == '__main__':
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="Autoencoder optimization.")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Activate the pruning feature. `PercentilePruner` stops unpromising "
             "trials at the early stages of training."
    )
    parser.add_argument(
        "--n_trials",
        "-n",
        action="store",
        type=int,
        default=10,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--max_epochs",
        "-m",
        action="store",
        type=int,
        default=2000,
        help="Number of epochs to run"
    )
    parser.add_argument(
        "--patience",
        "-p",
        action="store",
        type=int,
        default=100,
        help="Patience of the early stopping algorithm"
    )
    parser.add_argument(
        "--accelerator",
        "-a",
        action="store",
        help="Accelerator to use",
        choices=["cpu", "gpu"],
        default="cpu"
    )
    parser.add_argument(
        "--n_devices",
        action="store",
        default=1,
        type=int,
        help="Number of devices to use"
    )
    parser.add_argument(
        "--csv_path",
        "-c",
        action="store",
        default="PGA_LSTM/Datasets/FCR_2013_2018_Drivers.csv",
        help="Path of the csv data file"
    )
    parser.add_argument(
        "--work_dir",
        "-w",
        action="store",
        default=os.path.join("results", "autoencoder"),
        help="Path where logs and results will be stored"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Activate the profiler"
    )
    args = parser.parse_args()
    print("Program is run with ", args)
    logging.disable(logging.WARNING)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=1) if args.prune else optuna.pruners.NopPruner()
    study = optuna.create_study(storage="postgresql://davidenicoli@localhost:5432/optuna", study_name="autoencoder-optimization-maxim",
                                direction="maximize", pruner=pruner, load_if_exists=True)
    study.optimize(objective(args.accelerator, args.max_epochs, args.patience, args.csv_path, args.work_dir, args.n_devices, args.profile), n_trials=args.n_trials)
