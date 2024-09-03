import polars as pl
import numpy as np
import torch

from torch.utils.data import TensorDataset

from .tools import periodic_day, windowed
from .windowed import WindowedDataset


def make_autoencoder_dataset(drivers_table: pl.DataFrame, window_size: int):
    data = drivers_table.with_columns(periodic_day(pl.col("time")).alias("date_components")).unnest(
        "date_components").sort("time")
    return windowed(data.drop("time").to_numpy().astype(np.float32), window_size), None, windowed(
        data["time"].to_numpy(), window_size)

def make_autoencoder_split_dataset(x, t, window_size: int, test_frac: float, seed: int):
    windowed_dataset = WindowedDataset(TensorDataset(torch.from_numpy(x)), t)
    test_size = int(test_frac * len(windowed_dataset))
    indices = np.arange(len(windowed_dataset))
    np.random.seed(seed)
    np.random.shuffle(indices)
    test_indices = indices[:test_size]
    train_ds, test_ds = windowed_dataset.train_test_split(test_indices.copy())
    return train_ds, test_ds, windowed_dataset