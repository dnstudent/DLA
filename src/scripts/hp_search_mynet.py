import argparse
import logging
import os
from datetime import timedelta
from typing import Tuple

logging.disable(logging.CRITICAL)

import lightning as L
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping, Timer
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from rich.progress import track
from sklearn.model_selection import TimeSeriesSplit, KFold
from torch.utils import data as tdata

from src.datasets.fcr import spatiotemporal_split_dataset_v2
from src.datasets.tools import normalize_inputs
from src.models.regressors_v2 import MyNet
from src.tools.paths import ds_dir, drivers_path, embedding_path


def define_hparams(trial: optuna.Trial):
    apply_decay = trial.suggest_int("apply_decay", 0, 1)
    if apply_decay == 1:
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1, log=True)
    else:
        weight_decay = 0.

    hparams = {
        "weight_decay": weight_decay,
        "lr1": trial.suggest_float("lr1", 1e-4, 1e-2, log=True),
        "lr2": trial.suggest_float("lr2", 1e-4, 1e-2, log=True),
        "lr3": trial.suggest_float("lr3", 1e-4, 1e-2, log=True),
        "density_lambda": trial.suggest_float("density_lambda", 1e-2, 1e2, log=True),
        "hidden_dropout_rate": trial.suggest_float("hidden_dropout_rate", 0, 1),
        "input_dropout_rate": trial.suggest_float("input_dropout_rate", 0, 1),
        "z_dropout_rate": 0.0
    }
    return hparams

sets = {
    "big": {"forward_size": 12, "weather_embedding_size": 24, "hidden_size": 16},
    "small": {"forward_size": 3, "weather_embedding_size": 5, "hidden_size": 3},
    "medium": {"forward_size": 5, "weather_embedding_size": 8, "hidden_size": 5}
}

def prepare_their_data(x, w, y, train_idxs, val_idxs) -> Tuple[tdata.Dataset, tdata.Dataset]:
    x_train, w_train, y_train = x[train_idxs], w[train_idxs], y[train_idxs]
    x_val, w_val, y_val = x[val_idxs], w[val_idxs], y[val_idxs]
    (x_train, w_train, y_train), (x_val, w_val, y_val), _, _  = normalize_inputs([x_train, w_train, y_train], [x_val, w_val, y_val])
    train_ds = tdata.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(w_train),
                                   torch.from_numpy(y_train))
    val_ds = tdata.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(w_val), torch.from_numpy(y_val))
    return train_ds, val_ds

def objective(accelerator, max_epochs, patience, work_dir, n_devices, profile, log, seed, with_glm, net_size):
    L.seed_everything(seed)
    # Retrieving "train" dataset
    x, _, w, _, y, _, _, _ = spatiotemporal_split_dataset_v2(ds_dir("."), drivers_path(".", "fcr"), embedding_path(".", "fcr", None), with_glm, T_squared=False, z_poly=True)
    n_depth_features = x.shape[-1]
    n_weather_features = w.shape[-1]
    multiproc = n_devices > 1
    size_hparams = sets[net_size]
    def _fn(trial: optuna.Trial):
        cv = TimeSeriesSplit(n_splits=4, gap=90, test_size=20)
        # cv2 = KFold(n_splits=4, shuffle=True, random_state=seed)
        nrmse_t_scores = []
        nrmse_z_scores = []
        val_sizes = []
        hparams = define_hparams(trial)
        for i, (train_idxs, val_idxs) in track(enumerate(cv.split(x, y)), total=cv.n_splits, auto_refresh=False, transient=True, description=f"Trial {trial.number}"):
            train_ds, val_ds = prepare_their_data(x, w, y, train_idxs, val_idxs)

            trainer = L.Trainer(
                max_epochs=max_epochs,
                enable_checkpointing=False,
                accelerator=accelerator,
                devices=n_devices,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="valid/score/t", patience=patience, mode="max"),
                    Timer(duration=timedelta(minutes=5))
                ],
                logger=TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "logs"), version=f"fold_{i+1}") if log else False,
                log_every_n_steps=1,
                profiler=SimpleProfiler(dirpath=work_dir, filename="perf_logs") if profile else None,
                gradient_clip_val=1,
                gradient_clip_algorithm="norm"
            )
            model = MyNet(n_depth_features=n_depth_features, n_weather_features=n_weather_features, multiproc=multiproc, **hparams, **size_hparams)

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=256, shuffle=True),
                val_dataloaders=tdata.DataLoader(val_ds, batch_size=256, shuffle=False),
            )

            nrmse_t_scores.append(trainer.logged_metrics["valid/score/t"].item())
            nrmse_z_scores.append(trainer.logged_metrics["valid/score/z"].item())
            # trial.report(trainer.logged_metrics["valid/score/t"].item(), i)
            val_sizes.append(len(val_idxs))
            # if trial.should_prune():
            #     raise optuna.TrialPruned()
        return np.average(nrmse_t_scores, weights=val_sizes), np.average(nrmse_z_scores, weights=val_sizes)
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
        "--work_dir",
        action="store",
        default=os.path.join("results", "fcr", "mynet"),
        help="Path where logs and results will be stored"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Activate the profiler"
    )
    parser.add_argument(
        "--without_glm",
        action="store_true",
    )
    parser.add_argument(
        "--net_size",
        action="store",
        choices=["small", "medium", "big"],
        required=True
    )

if __name__ == '__main__':
    logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description="MyNet optimization")
    add_program_arguments(parser)
    args = parser.parse_args()
    print("Program is run with ", args)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=2, n_startup_trials=75) if args.prune else optuna.pruners.NopPruner()
    study_name = f"MyNet_{args.net_size}_test4y_kfold_fullparams_v2"
    study = optuna.create_study(
        storage="postgresql://davidenicoli@localhost:5432/dla_results",
        study_name=study_name,
        directions=["maximize", "maximize"],
        pruner=pruner,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=320)
    )

    workdir = os.path.join(args.work_dir, study.study_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    study.optimize(
        objective(args.accelerator, args.max_epochs, args.patience, workdir, args.n_devices, args.profile, args.log, seed=0, with_glm=not args.without_glm, net_size=args.net_size),
        n_trials=args.n_trials
    )
