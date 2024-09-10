import os
import logging
import argparse
logging.disable(logging.CRITICAL)

import lightning as L
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from rich.progress import track
from sklearn.model_selection import TimeSeriesSplit
from torch.utils import data as tdata

from src.datasets.fcr import spatiotemporal_split_dataset
from src.datasets.tools import normalize_inputs, their_edge_padding
from src.models import lstm
from src.models.callbacks import BestScore


def define_hparams(trial: optuna.Trial, physics_penalty):
    apply_decay = trial.suggest_int("apply_decay", 0, 1)
    if apply_decay == 1:
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1, log=True)
    else:
        weight_decay = 0
    # initial_lr = trial.suggest_float("initial_lr", 1e-5, 1e-1, log=True)
    density_lambda = trial.suggest_float("density_lambda", 1e-2, 1e2, log=True)
    hparams = {
        "weight_decay": weight_decay,
        # "initial_lr": initial_lr,
        "density_lambda": density_lambda,
    }
    if physics_penalty:
        physics_penalty_lambda = trial.suggest_float("physics_penalty_lambda", 1e-2, 1e2, log=True)
        hparams |= {"physics_penalty_lambda": physics_penalty_lambda}
    return hparams

def objective(model_name, accelerator, max_epochs, patience, ds_dir, embedding_dir, work_dir, n_devices, profile, log, seed, embedding_version):
    L.seed_everything(seed)
    if embedding_version:
        embedding_path = os.path.join(embedding_dir, f"{embedding_version}.csv")
    else:
        embedding_path = None
    # Retrieving "train" dataset
    x, _, _, _, y, _, _, _ = spatiotemporal_split_dataset(ds_dir, os.path.join(ds_dir, "FCR_2013_2018_Drivers.csv"), embedding_path)
    n_input_features = x.shape[-1]
    multiproc = n_devices > 1
    model_class = getattr(lstm, model_name)
    physics_penalty = "PGL" in model_name
    pad_size = 10
    def _fn(trial: optuna.Trial):
        cv = TimeSeriesSplit(n_splits=4, gap=31)
        nrmse_scores = []
        val_sizes = []
        hparams = define_hparams(trial, physics_penalty)
        for i, (train_idxs, val_idxs) in track(enumerate(cv.split(x, y)), total=cv.n_splits, auto_refresh=False, transient=True, description=f"Trial {trial.number}"):
            x_train, y_train = x[train_idxs], y[train_idxs]
            x_val, y_val = x[val_idxs], y[val_idxs]
            (x_train, y_train), (x_val, y_val), _, _ = normalize_inputs([x_train, y_train], [x_val, y_val])
            x_train, x_val = their_edge_padding(x_train, x_val, pad_steps=pad_size)

            train_ds = tdata.TensorDataset(torch.from_numpy(x_train), torch.empty((len(x_train), 1, 1)), torch.from_numpy(y_train))
            val_ds = tdata.TensorDataset(torch.from_numpy(x_val), torch.empty((len(x_val), 1, 1)), torch.from_numpy(y_val))

            best_score_logger = BestScore(monitor="valid/score/t", mode="max")
            trainer = L.Trainer(
                max_epochs=max_epochs,
                enable_checkpointing=False,
                accelerator=accelerator,
                devices=n_devices,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="valid/score/t", patience=patience, mode="max"),
                    best_score_logger
                ],
                logger=TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "logs"), version=f"fold_{i+1}") if log and i == 0 else False,
                log_every_n_steps=2,
                profiler=SimpleProfiler(dirpath=work_dir, filename="perf_logs") if profile else None,
                gradient_clip_val=1,
                gradient_clip_algorithm="norm"
            )
            model = model_class(n_input_features=n_input_features, skip_first=pad_size,  multiproc=multiproc, dropout_rate=0.2, initial_lr=1e-3, **hparams)

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=20, shuffle=True),
                val_dataloaders=tdata.DataLoader(val_ds, batch_size=256, shuffle=False),
            )
            # if trainer.logger:
            #     trainer.logger.log_hyperparams(params=hparams, metrics=trainer.callback_metrics)
            nrmse_scores.append(best_score_logger.best_score)
            val_sizes.append(len(x_val))
            trial.report(best_score_logger.best_score, i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        nrmse_score = np.average(nrmse_scores, weights=val_sizes)
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
        default=1000,
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


if __name__ == '__main__':
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="PGL LSTM optimization")
    add_program_arguments(parser)
    args = parser.parse_args()
    print("Program is run with ", args)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=2, n_startup_trials=25) if args.prune else optuna.pruners.NopPruner()
    study_name = f"{args.model_name}_{args.embedding_version}_test4y_timesplit_twoparams"
    study = optuna.create_study(
        storage="postgresql://davidenicoli@localhost:5432/dla_results",
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=20)
    )

    workdir = os.path.join(args.work_dir, study.study_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    study.optimize(
        objective(args.model_name, args.accelerator, args.max_epochs, args.patience, args.ds_dir, args.embedding_dir, workdir, args.n_devices, args.profile, args.log, seed=42, embedding_version=args.embedding_version),
        n_trials=args.n_trials
    )
