import os
import logging
import argparse

import lightning as L
import numpy as np
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from rich.progress import track
from sklearn.model_selection import KFold
from torch.utils import data as tdata

from src.datasets.fcr import spatiotemporal_split_dataset
from src.datasets.tools import normalize_inputs
from src.models import lstm


def define_hparams(trial: optuna.Trial, physics_penalty, ttoz_penalty):
    apply_decay = trial.suggest_int("apply_decay", 0, 1)
    if apply_decay == 1:
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    else:
        weight_decay = 0
    initial_lr = trial.suggest_float("initial_lr", 5e-4, 1e-1, log=True)
    lr_decay_rate = trial.suggest_float("lr_decay_rate", 0.1, 1, log=True)
    density_lambda = trial.suggest_float("density_lambda", 1e-1, 1e1, log=True)
    hparams = {
        "weight_decay": weight_decay,
        "initial_lr": initial_lr,
        "lr_decay_rate": lr_decay_rate,
        "density_lambda": density_lambda,
    }
    if physics_penalty:
        physics_penalty_lambda = trial.suggest_float("physics_penalty_lambda", 1e-2, 1e1, log=True)
        hparams |= {"physics_penalty_lambda": physics_penalty_lambda}
    if ttoz_penalty:
        ttoz_penalty_lambda = trial.suggest_float("ttoz_penalty_lambda", 1e-2, 1e1, log=True)
        hparams |= {"ttoz_penalty_lambda": ttoz_penalty_lambda}
    return hparams

def objective(model_name, accelerator, max_epochs, patience, ds_dir, embedding_dir, work_dir, n_devices, profile, log, test_size, shuffle_test, seed, embedding_version):
    L.seed_everything(seed)
    if embedding_version:
        embedding_path = os.path.join(embedding_dir, f"{embedding_version}.csv")
    else:
        embedding_path = None
    drivers_dir = os.path.join(ds_dir, "FCR_2013_2018_Drivers.csv")
    x, x_test, w, w_test, y, y_test, _, t_test = spatiotemporal_split_dataset(ds_dir, drivers_dir, embedding_path, test_size)
    n_input_features = x.shape[-1]
    n_initial_features = w.shape[-1]
    multiproc = n_devices > 1
    model_class = getattr(lstm, model_name)
    physics_penalty = "PGLLSTM" in model_name
    ttoz_penalty = model_name in ["MyPGALSTMLoss", "MyPGALSTMComb"]
    def _fn(trial: optuna.Trial):
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        nrmse_scores = []
        hparams = define_hparams(trial, physics_penalty, ttoz_penalty)
        for i, (train_idxs, val_idxs) in track(enumerate(cv.split(x, y)), total=cv.n_splits, auto_refresh=False, transient=True, description=f"Trial {trial.number}"):
            x_train, w_train, y_train = x[train_idxs], w[train_idxs], y[train_idxs]
            x_val, w_val, y_val = x[val_idxs], w[val_idxs], y[val_idxs]
            (x_train, w_train, y_train), (x_val, w_val, y_val), (x_means, w_means, y_means), (x_stds, w_stds, y_stds) = normalize_inputs([x_train, w_train, y_train], [x_val, w_val, y_val])
            py_means, py_stds = torch.from_numpy(y_means), torch.from_numpy(y_stds)
            pz_means, pz_stds = py_means[..., 1], py_stds[..., 1]
            pt_means, pt_stds = py_means[..., 0], py_stds[..., 0]
            train_ds = tdata.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(w_train), torch.from_numpy(y_train))
            val_ds = tdata.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(w_val), torch.from_numpy(y_val))

            trainer = L.Trainer(
                max_epochs=max_epochs,
                enable_checkpointing=False,
                accelerator=accelerator,
                devices=n_devices,
                enable_progress_bar=False,
                callbacks=[
                    EarlyStopping(monitor="valid/score/t", patience=patience, mode="max")
                ],
                logger=TensorBoardLogger(name=f'trial_{trial.number}', save_dir=os.path.join(work_dir, "logs"), version=f"fold_{i+1}") if log and i == 0 else False,
                log_every_n_steps=4,
                profiler=SimpleProfiler(dirpath=work_dir, filename="perf_logs") if profile else None,
            )
            model = model_class(n_input_features=n_input_features, **hparams, multiproc=multiproc, n_initial_features=n_initial_features, z_mean=pz_means, z_std=pz_stds, t_mean=pt_means, t_std=pt_stds, dropout_rate=0.2)

            trainer.fit(
                model,
                train_dataloaders=tdata.DataLoader(train_ds, batch_size=20, shuffle=True),
                val_dataloaders=tdata.DataLoader(val_ds, batch_size=128, shuffle=False),
            )
            # if trainer.logger:
            #     trainer.logger.log_hyperparams(params=hparams, metrics=trainer.callback_metrics)
            nrmse_scores.append(trainer.callback_metrics["valid/score/t"].item())
            trial.report(trainer.callback_metrics["valid/score/t"].item(), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        nrmse_score = np.mean(nrmse_scores)
        return nrmse_score
    return _fn


def add_program_arguments(parser):
    parser.add_argument(
        "--model_name",
        action="store",
        required=True,
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
        default=15000,
        help="Number of epochs to run"
    )
    parser.add_argument(
        "--patience",
        action="store",
        type=int,
        default=100,
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
    logging.disable(logging.WARNING)

    pruner = optuna.pruners.PercentilePruner(75.0, n_warmup_steps=0, n_startup_trials=20) if args.prune else optuna.pruners.NopPruner()
    study_name = f"{args.model_name}_{args.embedding_version}"
    study = optuna.create_study(
        storage="postgresql://davidenicoli@localhost:5432/dla_results",
        study_name=study_name,
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    workdir = os.path.join(args.work_dir, study.study_name)
    if not os.path.exists(workdir):
        os.makedirs(workdir)

    # embedding_version = args.embedding_version if args.embedding_version is not None else "latest"

    study.optimize(
        objective(args.model_name, args.accelerator, args.max_epochs, args.patience, args.ds_dir, args.embedding_dir, workdir, args.n_devices, args.profile, args.log, 0.6, False, 42, args.embedding_version),
        n_trials=args.n_trials
    )
