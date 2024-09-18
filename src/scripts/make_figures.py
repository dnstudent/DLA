import argparse
import logging
from os import PathLike
from pathlib import Path
from typing import List, Dict, Any, Optional

logging.disable(logging.CRITICAL)

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

from src.datasets.common import prepare_their_data
from src.datasets.fcr import spatiotemporal_split_dataset
from src.evaluation.analysis import test_rmse, test_physical_inconsistency, denorm
from src.models import lstm
from src.models.mcdropout import MCSampler
from src.models.training.v1 import train_or_load_best
from src.tools.paths import embedding_path, ds_dir, drivers_path
from src.evaluation.plot import make_figure_6


def add_program_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--wd",
        type=str,
        default=".",
        action="store"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/fcr/theirhp",
        action="store"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fcr",
        choices=["fcr", "mendota"],
        action="store"
    )
    parser.add_argument(
        "--without_glm",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=False
    )

def train_everything(model_classes: List[str], wd: PathLike[str], results_dir: PathLike[str], dataset: str, embedding_versions: List[Optional[str]], train_fracs: List[float], max_epochs: int, hparams: Dict[str, Dict[str, Any]], log, with_glm: bool, force_retrain: bool):
    test_results = []
    samples = {}
    final_models = {}
    for embedding_version in tqdm(embedding_versions, position=0):
        samples[embedding_version] = {}
        final_models[embedding_version] = {}
        # Retrieving "train" and test datasets
        x, x_test, w, w_test, y, y_test, _, _ = spatiotemporal_split_dataset(ds_dir(wd), drivers_path(wd, dataset), embedding_path(wd, dataset, embedding_version), with_glm)
        n_input_features = x.shape[-1]
        n_initial_features = w.shape[-1]
        for train_frac in tqdm(train_fracs, leave=False, position=1):
            samples[embedding_version][train_frac] = {}
            final_models[embedding_version][train_frac] = {}
            x_train, x_val, x_test1, w_train, w_val, w_test1, y_train, y_val, _, y_means, y_stds = prepare_their_data(x, w, y, x_test, w_test, y_test, train_frac, val_size=0.1, shuffle_train=True)
            train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(w_train), torch.from_numpy(y_train))
            val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(w_val), torch.from_numpy(y_val))
            for model_class in tqdm(model_classes, leave=False, position=2):
                # model = init_best_model(label, embedding_version, n_input_features)
                model = getattr(lstm, model_class)
                model, ckpt_path = train_or_load_best(results_dir, model, embedding_version, train_frac, hparams[model_class] | {"n_input_features": n_input_features}, ["without_glm"], train_ds, val_ds, log, max_epochs, force_retrain, trainer_kwargs=None)
                # model = c.load_from_checkpoint(ckpt_path, n_input_features=n_input_features, skip_first=pad_size, multiproc=multiproc)
                final_models[embedding_version][train_frac][model_class] = model
                mc_model = MCSampler(model, 100)
                sample = mc_model.sample(torch.from_numpy(x_test1), torch.from_numpy(w_test1), sample_size=100).detach().numpy()
                sample = denorm(sample, y_means, y_stds)
                samples[embedding_version][train_frac][model_class] = sample
                test_results.append({"model": model_class, "embedding_version": embedding_version, "train_frac": train_frac, "rmse": test_rmse(y_test, sample),
                               "physical_inconsistency": test_physical_inconsistency(sample), "ckpt_path": ckpt_path})
    return test_results, samples, final_models

their_names = {
    "TheirLSTM": "LSTM",
    "TheirPGL": "LSTM-PGL",
    "TheirPGA": "LSTM-PGA"
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_program_arguments(parser)
    args = parser.parse_args()
    model_labels = ["TheirLSTM", "TheirPGL", "TheirPGA"]

    hparams = {}
    for model_class in model_labels:
        hparams[model_class] = {"lr": 1e-3, "density_lambda": 1.0, "weight_decay": 0.05, "dropout_rate": 0.2, "multiproc": False}
    hparams["TheirPGL"]["physics_penalty_lambda"] = 1.0

    wd = Path(args.wd)
    results_dir = Path(args.results_dir)
    embedding_versions = ["orig", "3d_15r", None]
    embedding_paths = list(map(lambda v: embedding_path(wd, "fcr", v), embedding_versions))
    train_fracs = [.1, .2, .3, .4, .5, 1.]

    train_everything(model_labels, wd, results_dir, args.dataset, embedding_versions, train_fracs, max_epochs=5_000, hparams=hparams, log=args.log, with_glm=(not args.without_glm), force_retrain=args.force_retrain)
