import argparse
import logging
import os

import lightning as L
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from rich.progress import track
from torch.utils import data as tdata

from src.datasets.fcrdatasets import fcr_autoencoder_split_dataset
from src.datasets.transformers import StandardScaler, scale_wds
from src.models.autoencoder import LitTemporalAutoencoder
from src.validation.cv import MaxSizeKFold

WINDOW_SIZE = 7

def autoencoder(n_features, rnn_module, lr, optimizer_module, decoder_timesteps, representation_size):
    # rnn_module = getattr(nn, rnn_module_name)
    return LitTemporalAutoencoder(n_features=n_features,
                                  latent_dim=representation_size,
                                  num_layers=1,
                                  in_seq_length=WINDOW_SIZE,
                                  out_seq_length=decoder_timesteps,
                                  recurrent_module=rnn_module,
                                  optimizer_class=optimizer_module,
                                  lr=lr)

def define_hparams(trial: optuna.Trial):
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    optim_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta", "RMSprop"])
    return {"lr": lr, "optimizer_name": optim_name}

def objective(accelerator, max_epochs, patience, work_dir, n_devices, profile, recurrent_layer, decoder_timesteps, representation_size, windowed_dataset):
    n_features = windowed_dataset[:][0].shape[-1]
    def _fn(trial: optuna.Trial):
        cv = MaxSizeKFold(n_splits=3, max_val_frac=0.05)
        nrmse_scores = []
        dsizes = []
        hparams = define_hparams(trial)
        for i, (_, val_idxs) in track(enumerate(cv.split(windowed_dataset)), total=cv.n_splits, auto_refresh=False, transient=True, description=f"Trial {trial.number}"):
            train_ds, validation_ds = windowed_dataset.train_test_split(val_idxs.copy())
            scaler = StandardScaler()
            scaler.fit(train_ds.unique_entries(0))
            train_ds = scale_wds(scaler, train_ds)
            validation_ds = scale_wds(scaler, validation_ds)

            if profile:
                profiler = SimpleProfiler(dirpath=work_dir, filename="perf_logs")
            else:
                profiler = None
            logger = TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "logs"), version=f"fold_{i}") if i == 0 else None
            trainer = L.Trainer(
                max_epochs=max_epochs,
                enable_checkpointing=False,
                accelerator=accelerator,
                devices=n_devices,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="valid/nrmse", patience=patience, mode="max"),
                    LearningRateMonitor(logging_interval="epoch"),
                ],
                logger=logger,
                log_every_n_steps=10,
                profiler=profiler
            )
            model = autoencoder(n_features, getattr(torch.nn, recurrent_layer), hparams["lr"], getattr(torch.optim, hparams["optimizer_name"]), decoder_timesteps, representation_size)

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=64, shuffle=True),
                val_dataloaders=tdata.DataLoader(validation_ds, batch_size=128, shuffle=False),
            )
            trainer.logger.log_hyperparams(params=hparams, metrics=trainer.callback_metrics)
            dsizes.append(len(train_ds))
            nrmse_scores.append(trainer.callback_metrics["valid/nrmse"].item())
            trial.report(trainer.callback_metrics["valid/nrmse"].item(), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        nrmse_score = np.average(nrmse_scores, weights=dsizes)
        return nrmse_score
    return _fn

def add_program_arguments(parser):
    parser.add_argument(
        "--recurrent_layer",
        action="store",
        type=str,
        required=True,
        help="Recurrent layer to use in the autoencoder"
    )
    parser.add_argument(
        "--decoder_timesteps",
        action="store",
        required=True,
        type=int,
        help="Window size of decoder"
    )
    parser.add_argument(
        "--representation_size",
        action="store",
        required=True,
        type=int,
        help="Size of the compressed result"
    )
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
        default=15000,
        help="Number of epochs to run"
    )
    parser.add_argument(
        "--patience",
        "-p",
        action="store",
        type=int,
        default=1000,
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
        default=os.path.join("results", "fcr", "autoencoder"),
        help="Path where logs and results will be stored"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Activate the profiler"
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=42,
        help="Random number generator seed"
    )

if __name__ == '__main__':
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="Autoencoder optimization")
    add_program_arguments(parser)
    args = parser.parse_args()

    print("Program is run with ", args)
    logging.disable(logging.WARNING)

    train_ds, _, _ = fcr_autoencoder_split_dataset(args.csv_path, WINDOW_SIZE, test_frac=0.05, seed=args.seed)

    L.seed_everything(args.seed, workers=True)
    study_name = f"{args.recurrent_layer}_{args.decoder_timesteps}d_{args.representation_size}r"

    workdir = os.path.join(args.work_dir, study_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=0, n_startup_trials=10) if args.prune else optuna.pruners.NopPruner()
    study = optuna.create_study(
        storage="postgresql://davidenicoli@localhost:5432/optuna",
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )
    study.optimize(
        objective(args.accelerator, args.max_epochs, args.patience, workdir, args.n_devices, args.profile, args.recurrent_layer, args.decoder_timesteps, args.representation_size, train_ds),
        n_trials=args.n_trials
    )
