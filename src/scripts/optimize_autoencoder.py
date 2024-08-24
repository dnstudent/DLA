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
from rich.progress import track
from sklearn.model_selection import KFold
from torch.utils import data as tdata

from src.datasets.fcrdataset import fcr_dataset
from src.datasets.transformers import StandardScaler, scale_wds
from src.datasets.windowed import WindowedDataset
from src.models.autoencoder import LitTemporalAutoencoder

WINDOW_SIZE = 7
w = StandardScaler()

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
    rnn_module_name = trial.suggest_categorical("recurrent_layer", ["RNN", "LSTM"])
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
    optim_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta", "RMSprop"])
    return {"rnn_module_name": rnn_module_name, "lr": lr, "optimizer_name": optim_name}

def objective(accelerator, max_epochs, patience, csv_path, work_dir, n_devices, profile, decoder_timesteps, representation_size):
    X, y, t = fcr_dataset(csv_path, WINDOW_SIZE)
    n_features = X.shape[-1]-2 # Two of the columns are doy
    tads = WindowedDataset(tdata.TensorDataset(torch.from_numpy(X)), times=t)
    def _fn(trial: optuna.Trial):
        inner_cv = KFold(n_splits=3, shuffle=False)
        nrmse_scores = []
        r2_scores = []
        dsizes = []
        hparams = define_hparams(trial)
        for i, (_, val_idxs) in track(enumerate(inner_cv.split(tads)), total=inner_cv.n_splits, auto_refresh=False, transient=True, description=f"Trial {trial.number}"):
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
                    EarlyStopping(monitor="valid/nrmse", patience=patience, mode="max")
                ],
                logger=TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "lightning_logs"), version=f"fold_{i+1}"),
                log_every_n_steps=22,
                profiler=profiler
            )
            model = autoencoder(n_features, getattr(torch.nn, hparams["rnn_module_name"]), hparams["lr"], getattr(torch.optim, hparams["optimizer_name"]), decoder_timesteps, representation_size)

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=64, shuffle=True),
                val_dataloaders=tdata.DataLoader(val_ds, batch_size=128, shuffle=False),
            )
            trainer.logger.log_hyperparams(params=hparams, metrics=trainer.callback_metrics)
            dsizes.append(len(train_ds))
            nrmse_scores.append(trainer.callback_metrics["valid/nrmse"].item())
            r2_scores.append(trainer.callback_metrics["valid/r2"].item())
            # trial.report(trainer.callback_metrics["valid/nrmse"].item(), i)
            # if trial.should_prune():
            #     raise optuna.TrialPruned()
        nrmse_score = np.average(nrmse_scores, weights=dsizes)
        r2_score = np.average(r2_scores, weights=dsizes)
        return nrmse_score, r2_score
    return _fn


def add_program_arguments(parser):
    parser.add_argument(
        "--study_name",
        action="store",
        default="",
        type=str,
        help="Name of study to run"
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
        default=os.path.join("results"),
        help="Path where logs and results will be stored"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Activate the profiler"
    )

if __name__ == '__main__':
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="Autoencoder optimization")
    add_program_arguments(parser)
    args = parser.parse_args()
    if not args.study_name:
        study_p = f"{args.decoder_timesteps}d_{args.representation_size}r"
    else:
        study_p = args.study_name
    print("Program is run with ", args)
    logging.disable(logging.WARNING)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=0, n_startup_trials=16) if args.prune else optuna.pruners.NopPruner()
    study = optuna.create_study(
        storage="postgresql://davidenicoli@localhost:5432/optuna",
        study_name="_".join(["fcr_autoencoder_nodoy", study_p]),
        directions=["maximize", "maximize"],
        pruner=pruner,
        load_if_exists=True,
    )
    workdir = os.path.join(args.work_dir, study.study_name)

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    study.optimize(
        objective(args.accelerator, args.max_epochs, args.patience, args.csv_path, workdir, args.n_devices, args.profile, args.decoder_timesteps, args.representation_size),
        n_trials=args.n_trials
    )
