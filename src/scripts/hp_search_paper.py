import argparse
import logging
import os
from typing import Tuple
from datetime import timedelta

logging.disable(logging.CRITICAL)

import lightning as L
from lightning.pytorch.callbacks import Timer
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from rich.progress import track
from sklearn.model_selection import KFold, TimeSeriesSplit
from torch.utils import data as tdata

from src.datasets.fcr import spatiotemporal_split_dataset
from src.datasets.tools import normalize_inputs
from src.models import regressors
from src.tools.paths import ds_dir, drivers_path, embedding_path


def define_hparams(trial: optuna.Trial, physics_penalty):
    apply_decay = trial.suggest_int("apply_decay", 0, 1)
    if apply_decay == 1:
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1, log=True)
    else:
        weight_decay = 0
    hparams = {
        "weight_decay": weight_decay,
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "density_lambda": trial.suggest_float("density_lambda", 1e-2, 1e2, log=True),
    }
    if physics_penalty:
        hparams |= {"physics_penalty_lambda": trial.suggest_float("physics_penalty_lambda", 1e-2, 1e2, log=True)}
    return hparams

def prepare_their_data(x, y, train_idxs, val_idxs) -> Tuple[tdata.Dataset, tdata.Dataset]:
    x_train, y_train = x[train_idxs], y[train_idxs]
    x_val, y_val = x[val_idxs], y[val_idxs]
    (x_train, y_train), (x_val, y_val), (_, y_means), (_, y_stds)  = normalize_inputs([x_train, y_train], [x_val, y_val])
    y_train[..., 0] = y_train[..., 0]*y_stds[..., 0] + y_means[..., 0]
    y_val[..., 0] = y_val[..., 0]*y_stds[..., 0] + y_means[..., 0]

    train_ds = tdata.TensorDataset(torch.from_numpy(x_train), torch.empty((len(x_train), 1, 1)),
                                   torch.from_numpy(y_train))
    val_ds = tdata.TensorDataset(torch.from_numpy(x_val), torch.empty((len(x_val), 1, 1)), torch.from_numpy(y_val))
    return train_ds, val_ds

def objective(model_name, accelerator, max_epochs, patience, work_dir, n_devices, profile, log, seed, n, embedding_version, with_glm):
    # Retrieving "train" dataset
    x, _, _, _, y, _, _, _ = spatiotemporal_split_dataset(ds_dir("."), drivers_path(".", "fcr"), embedding_path(".", "fcr", embedding_version), with_glm, ordinal_day=True)
    n_input_features = x.shape[-1]
    multiproc = n_devices > 1
    model_class = getattr(regressors, model_name)
    physics_penalty = "PGL" in model_name
    def _fn(trial: optuna.Trial):
        cv = TimeSeriesSplit(n_splits=3, gap=90, test_size=32)
        nrmse_scores = []
        val_sizes = []
        hparams = define_hparams(trial, physics_penalty)
        for i, (train_idxs, val_idxs) in track(enumerate(cv.split(x, y)), total=cv.n_splits, auto_refresh=False, transient=True, description=f"Trial {trial.number}"):
            train_ds, val_ds = prepare_their_data(x, y, train_idxs, val_idxs)
            trainer = L.Trainer(
                max_epochs=max_epochs,
                enable_checkpointing=False,
                accelerator=accelerator,
                devices=n_devices,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="valid/score/t", patience=patience, mode="max"),
                    Timer(duration=timedelta(minutes=3.5), interval="epoch")
                ],
                logger=TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "logs"), version=f"fold_{i+1}") if log and i == 0 else False,
                log_every_n_steps=1,
                profiler=SimpleProfiler(dirpath=work_dir, filename="perf_logs") if profile else None,
                gradient_clip_val=1,
                gradient_clip_algorithm="norm"
            )
            model = model_class(n_input_features=n_input_features,  multiproc=multiproc, dropout_rate=0.2, temperature_lambda=1.0, **hparams)

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=20, shuffle=True),
                val_dataloaders=tdata.DataLoader(val_ds, batch_size=256, shuffle=False),
            )

            nrmse_scores.append(trainer.logged_metrics["valid/score/t"].item())
            trial.report(trainer.logged_metrics["valid/score/t"].item(), i)
            val_sizes.append(len(val_ds))
            if trial.should_prune():
                raise optuna.TrialPruned()
        nrmse_score = np.average(nrmse_scores[1:], weights=val_sizes[1:])
        return nrmse_score
    return _fn

def add_program_arguments(parser):
    parser.add_argument(
        "--model_name",
        action="store",
        required=True,
        choices=["TheirLSTM", "TheirPGL", "TheirPGA"],
        type=str,
        help="Name of the model to optimize"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Activate the pruning feature. `PercentilePruner` stops unpromising "
             "trials at the early stages of training."
    )
    parser.add_argument(
        "--n_trials",
        action="store",
        type=int,
        required=True,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--max_epochs",
        action="store",
        type=int,
        default=30_000,
        help="Number of epochs to run"
    )
    parser.add_argument(
        "--patience",
        action="store",
        type=int,
        default=200,
        help="Patience of the early stopping algorithm"
    )
    parser.add_argument(
        "--accelerator",
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
        "--ds_dir",
        action="store",
        default=os.path.join("PGA_LSTM", "Datasets"),
        help="Directory containing the weather drivers csv"
    )
    parser.add_argument(
        "--embedding_dir",
        action="store",
        default=os.path.join("data", "fcr", "encoded_features"),
    )
    parser.add_argument(
        "--work_dir",
        action="store",
        default=os.path.join("results", "fcr", "lstm"),
        help="Path where logs and results will be stored"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Activate the profiler"
    )
    parser.add_argument(
        "--embedding_version",
        action="store",
        default=None,
        type=str,
        help="Embedding version to use in the form {n}d_{n}r"
    )
    parser.add_argument(
        "--without_glm",
        action="store_true",
    )

if __name__ == '__main__':
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="LSTM optimization")
    add_program_arguments(parser)
    args = parser.parse_args()
    print("Program is run with ", args)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=2, n_startup_trials=40) if args.prune else optuna.pruners.NopPruner()
    study_name = f"{args.model_name}_{args.embedding_version}_test4y_timesplit2_fullparams"
    study = optuna.create_study(
        storage="postgresql://davidenicoli@localhost:5432/dla_results",
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=100)
    )

    workdir = os.path.join(args.work_dir, study.study_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    study.optimize(
        objective(args.model_name, args.accelerator, args.max_epochs, args.patience, workdir, args.n_devices, args.profile, args.log, seed=-1, embedding_version=args.embedding_version, with_glm=not args.without_glm),
        n_trials=args.n_trials
    )
