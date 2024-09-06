import os
import logging
import argparse

import lightning as L
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, Timer
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from rich.progress import track
from sklearn.model_selection import KFold
from torch.utils import data as tdata

from src.datasets.fcr import spatiotemporal_split_dataset_v2
from src.datasets.tools import normalize_inputs
from src.models.lstm_v2 import SmallRegressor
from src.models.callbacks import BestScore

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield err, out

def suppressed_fn(fn):
    def suppressed(*args, **kwargs):
        with suppress_stdout_stderr():
            return fn(*args, **kwargs)
    return suppressed


def define_hparams(trial: optuna.Trial):
    apply_decay = trial.suggest_int("apply_decay", 0, 1)
    if apply_decay == 1:
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)
    else:
        weight_decay = 0
    initial_lr1 = trial.suggest_float("initial_lr1", 1e-4, 5e-1, log=True)
    initial_lr2 = trial.suggest_float("initial_lr2", 1e-4, 5e-1, log=True)
    initial_lr3 = trial.suggest_float("initial_lr3", 1e-4, 5e-1, log=True)
    lr_decay_rate = trial.suggest_float("lr_decay_rate", 0.1, 1, log=True)
    density_lambda = trial.suggest_float("density_lambda", 1e-2, 1e2, log=True)
    forward_size = trial.suggest_int("forward_size", 1, 10)
    weather_embedding_size = trial.suggest_int("weather_embedding_size", 1, 15)
    n_delta_layers = trial.suggest_int("n_delta_layers", 1, 5)
    return {
        "weight_decay": weight_decay,
        "initial_lr1": initial_lr1,
        "initial_lr2": initial_lr2,
        "initial_lr3": initial_lr3,
        "lr_decay_rate": lr_decay_rate,
        "density_lambda": density_lambda,
        "forward_size": forward_size,
        "weather_embedding_size": weather_embedding_size,
        "n_delta_layers": n_delta_layers,
    }

# @suppressed_fn
def objective(max_epochs, patience, ds_dir, work_dir, profile, log, test_size):
    drivers_dir = os.path.join(ds_dir, "FCR_2013_2018_Drivers.csv")
    x, x_test, w, w_test, y, y_test, _, t_test = spatiotemporal_split_dataset_v2(ds_dir, drivers_dir, None, test_size)
    n_depth_features = x.shape[-1]
    n_weather_features = w.shape[-1]
    multiproc = False
    def _fn(trial: optuna.Trial):
        cv = KFold(n_splits=4, shuffle=False, random_state=None)
        nrmse_scores = []
        hparams = define_hparams(trial)
        for i, (train_idxs, val_idxs) in track(enumerate(cv.split(x, y)), total=cv.n_splits, auto_refresh=False, transient=True, description=f"Trial {trial.number}"):
            x_train, w_train, y_train = x[train_idxs], w[train_idxs], y[train_idxs]
            x_val, w_val, y_val = x[val_idxs], w[val_idxs], y[val_idxs]
            (x_train, w_train, y_train), (x_val, w_val, y_val), (x_means, w_means, y_means), (x_stds, w_stds, y_stds) = normalize_inputs([x_train, w_train, y_train], [x_val, w_val, y_val])

            train_ds = tdata.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(w_train), torch.from_numpy(y_train))
            val_ds = tdata.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(w_val), torch.from_numpy(y_val))
            best_score_logger = BestScore(monitor="valid/score/t", mode="max")
            trainer = L.Trainer(
                max_epochs=max_epochs,
                enable_checkpointing=False,
                accelerator="cpu",
                devices=1,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="valid/score/t", patience=patience, mode="max"),
                    best_score_logger
                ],
                logger=TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "logs"), version=f"fold_{i+1}") if log and i == 0 else False,
                log_every_n_steps=4,
                profiler=SimpleProfiler(dirpath=work_dir, filename="perf_logs") if profile else None,
            )
            model = SmallRegressor(n_depth_features=n_depth_features, **hparams, multiproc=multiproc, n_weather_features=n_weather_features, dropout_rate=0.2)

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=20, shuffle=True),
                val_dataloaders=tdata.DataLoader(val_ds, batch_size=128, shuffle=False),
            )
            nrmse_scores.append(best_score_logger.best_score)
            trial.report(best_score_logger.best_score, i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        nrmse_score = np.mean(nrmse_scores)
        return nrmse_score
    return _fn


def add_program_arguments(parser):
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
        default=50_000,
        help="Number of epochs to run"
    )
    parser.add_argument(
        "--patience",
        action="store",
        type=int,
        default=8_000,
        help="Patience of the early stopping algorithm"
    )
    parser.add_argument(
        "--ds_dir",
        action="store",
        default=os.path.join("PGA_LSTM", "Datasets"),
        help="Directory containing the weather drivers csv"
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


if __name__ == '__main__':
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="PGL LSTM optimization")
    add_program_arguments(parser)
    args = parser.parse_args()
    print("Program is run with ", args)
    logging.disable(logging.WARNING)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=0, n_startup_trials=20) if args.prune else optuna.pruners.NopPruner()
    study_name = f"SmallPGA"
    study = optuna.create_study(
        storage="postgresql://davidenicoli@localhost:5432/dla_results",
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.RandomSampler()
    )

    workdir = os.path.join(args.work_dir, study.study_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    study.optimize(
        objective(args.max_epochs, args.patience, args.ds_dir, workdir, args.profile, args.log, test_size=0.2),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
