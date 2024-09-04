import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from .tools import periodic_day, windowed, density
from .windowed import WindowedDataset


def make_autoencoder_dataset_old(drivers_table: pl.DataFrame, window_size: int):
    data = drivers_table.with_columns(periodic_day(pl.col("date")).alias("date_components")).unnest(
        "date_components").sort("date")
    return windowed(data.drop("date").to_numpy().astype(np.float32), window_size), None, windowed(
        data["date"].to_numpy(), window_size)

def make_autoencoder_dataset(drivers_reader):
    def _fn(drivers_table: pl.DataFrame, window_size: int):
        if type(drivers_table) is str:
            drivers_table = drivers_reader(drivers_table)
        data = drivers_table.with_columns(periodic_day(pl.col("date")).alias("date_components")).unnest(
            "date_components").sort("date")
        return windowed(data.drop("date").to_numpy().astype(np.float32), window_size), None, windowed(
            data["date"].to_numpy(), window_size)
    return _fn

def make_autoencoder_split_dataset_old(x, t, test_frac: float, seed: int, shuffle: bool):
    windowed_dataset = WindowedDataset(TensorDataset(torch.from_numpy(x)), t)
    test_size = int(test_frac * len(windowed_dataset))
    indices = np.arange(len(windowed_dataset))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    test_indices = indices[-test_size:]
    train_ds, test_ds = windowed_dataset.train_test_split(test_indices.copy())
    return train_ds, test_ds, windowed_dataset

def make_autoencoder_split_dataset(autoencoder_dataset_maker):
    def _fn(drivers_path: str, window_size: int, test_frac: float, seed: int, shuffle: bool):
        x, _, t = autoencoder_dataset_maker(drivers_path, window_size)
        windowed_dataset = WindowedDataset(TensorDataset(torch.from_numpy(x)), t)
        test_size = int(test_frac * len(windowed_dataset))
        indices = np.arange(len(windowed_dataset))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        test_indices = indices[-test_size:]
        train_ds, test_ds = windowed_dataset.train_test_split(test_indices.copy())
        return train_ds, test_ds, windowed_dataset
    return _fn

def make_spatiotemporal_dataset(full_table_loader, depth_steps):
    def _fn(ds_dir, embedded_features_csv_path):
        table = full_table_loader(ds_dir, embedded_features_csv_path)
        table = table.filter(pl.col("temp").is_not_null()).sort("date", "depth").drop("date")
        return (
            table.drop(["temp"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, len(table.columns) - 1)),
            table.select(["temp"]).with_columns(density(pl.col("temp")).alias("density")).select(["temp", "density"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, 2))
        )
    return _fn

def make_spatiotemporal_split_dataset(spatiotemporal_dataset_maker, depth_steps):
    def _fn(ds_dir, embedded_features_csv_path, test_size, seed=42, shuffle=False):
        X, y = spatiotemporal_dataset_maker(ds_dir, embedded_features_csv_path)
        return train_test_split(X, y, test_size=test_size, random_state=seed if shuffle else None, shuffle=shuffle)
    return _fn