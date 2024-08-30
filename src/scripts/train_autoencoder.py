import argparse
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn, optim

from src.datasets.fcrdatasets import FCRAutoencoderDataModule
from src.models.autoencoder import LitTemporalAutoencoder


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--latent_size",
        action="store",
        type=int,
        required=True,
        help="Latent size of the autoencoder"
    )
    parser.add_argument(
        "--output_timesteps",
        action="store",
        type=int,
        required=True,
        help="Output timesteps of the autoencoder that are confronted with original data"
    )
    parser.add_argument(
        "--lr",
        action="store",
        type=float,
        required=True,
        help="Learning rate of the autoencoder"
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--validation_size",
        action="store",
        type=float,
        default=0.05,
        help="Fraction of data to use for validation. Training size may vary"
    )
    parser.add_argument(
        "--recurrent_module",
        action="store",
        type=str,
        required=True,
        help="Recurrent module to use"
    )
    parser.add_argument(
        "--optimizer",
        action="store",
        type=str,
        required=True,
        help="Optimizer of the autoencoder"
    )
    parser.add_argument(
        "--working_dir",
        action="store",
        type=str,
        default=os.path.join("results", "autoencoder")
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    L.seed_everything(args.seed, workers=True)
    dm = FCRAutoencoderDataModule("PGA_LSTM/Datasets/FCR_2013_2018_Drivers.csv", 7, 32, args.validation_size)

    recurrent_module=getattr(nn, args.recurrent_module)
    optimizer = getattr(optim, args.optimizer)
    autoencoder = LitTemporalAutoencoder(
        n_features=dm.n_features,
        latent_dim=args.latent_size,
        num_layers=1,
        in_seq_length=dm.n_timesteps,
        out_seq_length=args.output_timesteps,
        recurrent_module=recurrent_module,
        optimizer_class=optimizer,
        lr=args.lr
    )

    wd = Path(args.working_dir)
    if not wd.exists():
        wd.mkdir(parents=True)
    project_tag = f"{args.output_timesteps}d_{args.latent_size}r"
    trainer = L.Trainer(
        default_root_dir=wd / project_tag,
        max_epochs=15000,
        enable_checkpointing=True,
        accelerator="cpu",
        logger=TensorBoardLogger(save_dir=wd / project_tag / "logs", name=f"{args.optimizer}_{args.recurrent_module}"),
        callbacks=[
            EarlyStopping(monitor="valid/nrmse", patience=100, mode="max"),
            ModelCheckpoint(monitor="valid/nrmse", mode="max")
        ]
    )
    trainer.fit(autoencoder, datamodule=dm)