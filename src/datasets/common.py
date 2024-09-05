import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from .tools import periodic_day, windowed, density, split_temporal_rolling
from .windowed import WindowedDataset


def make_autoencoder_dataset(drivers_reader):
    def _fn(drivers_table: pl.DataFrame, window_size: int):
        if type(drivers_table) is str:
            drivers_table = drivers_reader(drivers_table)
        data = drivers_table.with_columns(periodic_day(pl.col("date")).alias("date_components")).unnest(
            "date_components").sort("date")
        return windowed(data.drop("date").to_numpy().astype(np.float32), window_size), None, windowed(
            data["date"].to_numpy(), window_size)
    return _fn

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

def make_spatiotemporal_dataset(full_table_loader, drivers_reader, depth_steps, time_steps):
    def _fn(ds_dir, drivers_path, embedded_features_csv_path):
        table = full_table_loader(ds_dir, embedded_features_csv_path)
        table = table.filter(pl.col("temp").is_not_null()).sort("date", "depth")
        drivers = (
            drivers_reader(drivers_path)
            .with_columns(date_components=periodic_day(pl.col("date")))
            .unnest("date_components")
            .select("date", features=pl.struct(cs.all()))
            .group_by_dynamic(index_column="date", period="7d", every="1d", include_boundaries=False, label="right", closed="right")
            .agg(pl.col("features"), n=pl.len())
            .filter(pl.col("n") == time_steps)
            .drop("n")
            .join(table, on="date", how="semi")
            .sort("date")
        )
        table = table.join(drivers, on="date", how="semi")
        return (
            # x
            table.drop(["date", "temp"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, len(table.columns) - 2)),
            # w
            drivers.drop("date").explode("features").unnest("features").drop("date").to_numpy().astype(np.float32).reshape((len(drivers), time_steps, -1)),
            # y
            table.select(["temp"]).with_columns(density(pl.col("temp")).alias("density")).select(["temp", "density"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, 2)),
            # t
            drivers["date"].to_numpy()
        )
    return _fn

def make_spatiotemporal_dataset_v2(full_table_loader, drivers_reader, depth_steps, time_steps):
    def _fn(ds_dir, drivers_path, embedded_features_csv_path):
        table = full_table_loader(ds_dir, embedded_features_csv_path)
        table = table.filter(pl.col("temp").is_not_null()).sort("date", "depth")
        drivers = (
            drivers_reader(drivers_path)
            .with_columns(date_components=periodic_day(pl.col("date")))
            .unnest("date_components")
            .select("date", features=pl.struct(cs.all()))
            .group_by_dynamic(index_column="date", period="7d", every="1d", include_boundaries=False, label="right", closed="right")
            .agg(pl.col("features"), n=pl.len())
            .filter(pl.col("n") == time_steps)
            .drop("n")
            .join(table, on="date", how="semi")
            .sort("date")
        )
        table = table.join(drivers, on="date", how="semi")
        return (
            # d
            table.select("depth", "glm_temp").to_numpy().astype(np.float32).reshape((-1, depth_steps, 2)),
            # w
            drivers.drop("date").explode("features").unnest("features").drop("date").to_numpy().astype(np.float32).reshape((len(drivers), time_steps, -1)),
            # y
            table.select(["temp"]).with_columns(density(pl.col("temp")).alias("density")).select(["temp", "density"]).to_numpy().astype(np.float32).reshape((-1, depth_steps, 2)),
            # t
            drivers["date"].to_numpy()
        )
    return _fn

def make_spatiotemporal_split_dataset_pedantic(spatiotemporal_dataset_maker):
    def _fn(ds_dir, drivers_path, embedded_features_csv_path, test_size):
        x, w, y, t = spatiotemporal_dataset_maker(ds_dir, drivers_path, embedded_features_csv_path)
        (t_train, t_test), (w_train, w_test), (x_train, x_test), (y_train, y_test) = split_temporal_rolling(t, w, x, y, right_frac=test_size)
        return x_train, x_test, w_train, w_test, y_train, y_test, t_train, t_test
    return _fn

def make_spatiotemporal_split_dataset(spatiotemporal_dataset_maker):
    def _fn(ds_dir, drivers_path, embedded_features_csv_path, test_size, random_state=None, shuffle=False):
        x, w, y, t = spatiotemporal_dataset_maker(ds_dir, drivers_path, embedded_features_csv_path)
        return train_test_split(x, w, y, t, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return _fn