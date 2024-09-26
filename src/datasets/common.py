from os import PathLike
from typing import Union
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from .tools import periodic_day, windowed, density, take_frac, normalize_inputs
from .windowed import WindowedDataset


def make_autoencoder_dataset(drivers_reader):
    def _fn(drivers_table: pl.DataFrame, window_size: int, ordinal_day):
        if type(drivers_table) is str:
            drivers_table = drivers_reader(drivers_table)
        data = drivers_table.sort("date")
        if ordinal_day:
            data = data.with_columns(day=pl.col("date").dt.ordinal_day())
        else:
            data = data.with_columns(periodic_day(pl.col("date")).alias("date_components")).unnest(
            "date_components")
        return windowed(data.drop("date").to_numpy().astype(np.float32), window_size), None, windowed(
            data["date"].to_numpy(), window_size)
    return _fn

def make_autoencoder_split_dataset(autoencoder_dataset_maker):
    def _fn(drivers_path: str, window_size: int, test_frac: float, seed: int, shuffle: bool, ordinal_day=True):
        x, _, t = autoencoder_dataset_maker(drivers_path, window_size, ordinal_day)
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
    def _fn(ds_dir: PathLike, drivers_path: PathLike, embedded_features_csv_path: PathLike, with_glm: bool, ordinal_day: bool):
        table = full_table_loader(ds_dir, embedded_features_csv_path, with_glm, ordinal_day)
        table = table.filter(pl.col("temp").is_not_null()).sort("date", "depth")
        drivers = (
            drivers_reader(drivers_path)
            .with_columns(date_components=periodic_day(pl.col("date")))
            .unnest("date_components")
            .select("date", features=pl.struct(cs.all()))
            .group_by_dynamic(index_column="date", period=f"{time_steps}d", every="1d", include_boundaries=False, label="right", closed="right")
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
    def _fn(ds_dir, drivers_path, embedded_features_csv_path, with_glm, T_squared, z_poly):
        table = full_table_loader(ds_dir, embedded_features_csv_path, with_glm)
        table = table.filter(pl.col("temp").is_not_null()).sort("date", "depth")
        drivers = (
            drivers_reader(drivers_path)
            .with_columns(date_components=periodic_day(pl.col("date")))
            .unnest("date_components")
            .select("date", features=pl.struct(cs.all()))
            .group_by_dynamic(index_column="date", period=f"{time_steps}d", every="1d", include_boundaries=False, label="right", closed="right")
            .agg(pl.col("features"), n=pl.len())
            .filter(pl.col("n") == time_steps)
            .drop("n")
            .join(table, on="date", how="semi")
            .sort("date")
        )
        table = table.join(drivers, on="date", how="semi")
        y: pl.DataFrame = table.select(["temp"]).with_columns(density(pl.col("temp")).alias("density")).select(["temp", "density"])
        if T_squared:
            y = y.with_columns(temp = pl.col("temp")**2)
        y = y.to_numpy().astype(np.float32).reshape((-1, depth_steps, 2))

        d = table.select("depth", "glm_temp")
        if z_poly:
            d = d.with_columns(depth_third=np.cbrt(pl.col("depth")), depth3=pl.col("depth")**3).select("depth", "glm_temp", "depth_third", "depth3")
        d = d.to_numpy().astype(np.float32).reshape((len(y), depth_steps, -1))

        return (
            # x
            d,
            # w
            drivers.drop("date").explode("features").unnest("features").drop("date").to_numpy().astype(np.float32).reshape((len(drivers), time_steps, -1)),
            # y
            y,
            # t
            drivers["date"].to_numpy()
        )
    return _fn

# def make_spatiotemporal_split_dataset_pedantic(spatiotemporal_dataset_maker):
#     def _fn(ds_dir, drivers_path, embedded_features_csv_path, test_size):
#         x, w, y, t = spatiotemporal_dataset_maker(ds_dir, drivers_path, embedded_features_csv_path)
#         (t_train, t_test), (w_train, w_test), (x_train, x_test), (y_train, y_test) = split_temporal_rolling(t, w, x, y, right_frac=test_size)
#         return x_train, x_test, w_train, w_test, y_train, y_test, t_train, t_test
#     return _fn

def make_spatiotemporal_split_dataset(spatiotemporal_dataset_maker, split):
    def _fn(ds_dir, drivers_path, embedded_features_csv_path, with_glm, ordinal_day, **kwargs):
        x, w, y, t = spatiotemporal_dataset_maker(ds_dir, drivers_path, embedded_features_csv_path, with_glm, ordinal_day, **kwargs)
        return x[:split], x[split:], w[:split], w[split:], y[:split], y[split:], t[:split], t[split:]
    return _fn

def prepare_their_data(x, w, y, x_test, w_test, y_test, train_size, val_size, shuffle_train, random_state, denorm_T=True):
    # Taking the actual train dataset as a subset
    x_train, w_train, y_train = take_frac(x, w, y, axis=0, frac=train_size, shuffle=shuffle_train, random_state=random_state)
    # Splitting in training and validation
    x_train, x_val, w_train, w_val, y_train, y_val = train_test_split(x_train, w_train, y_train, test_size=val_size, shuffle=False)
    # Normalizing the datasets
    (x_train, w_train, y_train), (x_val, w_val, y_val), (x_test, w_test, y_test), (x_means, w_means, y_means), (x_stds, w_stds, y_stds) = normalize_inputs(
        [x_train, w_train, y_train], [x_val, w_val, y_val], [x_test, w_test, y_test])
    # Temperatures are not normalized
    if denorm_T:
        y_train[..., 0] = y_train[..., 0]*y_stds[..., 0] + y_means[..., 0]
        y_val[..., 0] = y_val[..., 0]*y_stds[..., 0] + y_means[..., 0]
        y_test[..., 0] = y_test[..., 0]*y_stds[..., 0] + y_means[..., 0]
    # Padding the datasets
    # x_train, x_val, x_test = their_edge_padding(x_train, x_val, x_test, pad_steps=pad_size, axis=1)
    return x_train, x_val, x_test, w_train, w_val, w_test, y_train, y_val, y_test, y_means, y_stds